// #include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "params.h"
#include "timer.h"
#include "utilities.h"

typedef struct {
    float *x_res;
    float *x;
    float *x2;
    float *k_cache;
    float *v_cache;
    float *q;
    float *kq;
    float *z;
    float *logits;
} State;

typedef struct {
    float *token_embedding_table;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *rms_att_w;
    float *rms_ffn_w;
    float *rms_final_w;
    float *w1;
    float *w2;
    float *w3;
    float *wcls;
} Weights;

typedef struct {
    int embed_dim;   // transformer dimension
    int hidden_dim;  // for ffn layers
    int n_layers;    // number of layers
    int n_heads;     // number of query heads
    int n_kv_heads;  // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;  // vocabulary size, usually 256 (byte-level)
    int seq_len;     // max sequence length
} Config;

typedef struct {
    State *state;
    Weights *weights;
    Config *config;
} Transformer;

void print(float *x, int d, int n) {
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << x[i * n + j] << " ";
        }
        std::cout << std::endl;
        // std::cout << std::endl;
        // return;
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

// TODO: simd :)
void inplace_swish(float *x, int d) {
#pragma omp parallel for
    for (int i = 0; i < d; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

void inplace_rope(float *x, int pos, int dim) {
    // TODO: maybe can precalculate power (theta)
    for (int i = 0; i < dim; i += 2) {
        // is there supposed to be a -2 in the exponent?
        float theta = powf(10000.0f, i / (float)dim);
        float val = pos / theta;
        x[i] = x[i] * cosf(val) - x[i + 1] * sinf(val);
        x[i + 1] = x[i + 1] * cosf(val) - x[i] * sinf(val);
    }
}

void rms_norm(float *x, float *w, float *o, int d) {
    float rms = 0.0;
#pragma omp parallel for reduction(+ : rms)
    for (int i = 0; i < d; i++)
        rms += x[i] * x[i];

    rms = sqrt(rms / d + 1e-5f);

#pragma omp parallel for
    for (int i = 0; i < d; i++)
        o[i] = x[i] * w[i] / rms;
}

void inplace_softmax(float *x, int d) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < d; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < d; i++) {
        x[i] /= sum;
    }
}

// w (d, n) @ x (n) = xout (d)
void b_matvec(float *w, float *x, float *x_out, int d, int n) {
    int block_size = 4;
    using blocked_matrix_t = float(&)[d / block_size][n / block_size][block_size][block_size];
    blocked_matrix_t w_m = reinterpret_cast<blocked_matrix_t>(*w);

#pragma omp parallel for
    for (int i = 0; i < d / block_size; i++) {
        float32x4_t v_out = vdupq_n_f32(0.0);
        for (int j = 0; j < n / block_size; j++) {
            for (int ii = 0; ii < block_size; ii++) {
                float32x4_t v1 = vld1q_f32((float *)&w_m[i][j][ii]);
                float32x4_t v2 = vdupq_n_f32(x[j * block_size + ii]);
                v_out = vfmaq_f32(v_out, v1, v2);
            }
        }
        vst1q_f32(&x_out[i * block_size], v_out);
    }
}

// w (d, n) @ x (n) + xout (d) = xout (d)
// untested
void b_matvecadd(float *w, float *x, float *x_out, int d, int n) {
    int block_size = 4;
    using blocked_matrix_t = float(&)[d / block_size][n / block_size][block_size][block_size];
    blocked_matrix_t w_m = reinterpret_cast<blocked_matrix_t>(*w);

#pragma omp parallel for
    for (int i = 0; i < d / block_size; i++) {
        float32x4_t v_out = vld1q_f32(&x_out[i * block_size]);
        for (int j = 0; j < n / block_size; j++) {
            for (int ii = 0; ii < block_size; ii++) {
                float32x4_t v1 = vld1q_f32((float *)&w_m[i][j][ii]);
                float32x4_t v2 = vdupq_n_f32(x[j * block_size + ii]);
                v_out = vfmaq_f32(v_out, v1, v2);
            }
        }
        vst1q_f32(&x_out[i * block_size], v_out);
    }
}

// w (d, n) @ x (n) * xout (d) = xout (d)
// untested
void b_matvecmul(float *w, float *x, float *x_out, int d, int n) {
    int block_size = 4;
    using blocked_matrix_t = float(&)[d / block_size][n / block_size][block_size][block_size];
    blocked_matrix_t w_m = reinterpret_cast<blocked_matrix_t>(*w);

#pragma omp parallel for
    for (int i = 0; i < d / block_size; i++) {
        float32x4_t v_out = vdupq_n_f32(0.0);
        for (int j = 0; j < n / block_size; j++) {
            for (int ii = 0; ii < block_size; ii++) {
                float32x4_t v1 = vld1q_f32((float *)&w_m[i][j][ii]);
                float32x4_t v2 = vdupq_n_f32(x[j * block_size + ii]);
                v_out = vfmaq_f32(v_out, v1, v2);
            }
        }
        v_out = vmulq_f32(v_out, vld1q_f32(&x_out[i * block_size]));
        vst1q_f32(&x_out[i * block_size], v_out);
    }
}

/*
void matmul_cblas(const float *A, const float *B, float *C, const int m, const int n) {
    // Leading dimensions of the matrices
    int lda = m;  // Since A is m-by-k and not transposed
    int incX = 1;
    int incY = 1;

    // Scalar multipliers (alpha and beta)
    float alpha = 1.0;  // Multiplier for the matrices A and B
    float beta = 0.0;   // Multiplier for matrix C

    // Perform the matrix multiplication: C = alpha*A*B + beta*C
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                m, n,
                alpha, A, lda,
                B, incX,
                beta, C, incY);
}
*/

// TODO: current implementation is does not support GQA
void kq_mul_softmax_fused(float *q, float *k_cache, float *kq, int n_heads, int n_kv_heads, int pos, int kqv_dim) {
    // q heads per kv head
    // int kv_head_ratio = n_heads / n_kv_heads;

    using q_matrix_t = float(&)[n_heads][kqv_dim];
    using k_matrix_t = float(&)[pos + 1][n_heads][kqv_dim];
    using kq_matrix_t = float(&)[n_heads][pos + 1];

    q_matrix_t q_m = reinterpret_cast<q_matrix_t>(*q);
    k_matrix_t k_m = reinterpret_cast<k_matrix_t>(*k_cache);
    kq_matrix_t kq_m = reinterpret_cast<kq_matrix_t>(*kq);

    float sqrt_d = sqrtf((float)kqv_dim);

    // TODO: add simd
    // no sequential i/o on k_cache but fused operation (need to test performance)

#pragma omp parallel for
    for (int i = 0; i < n_heads; i++) {
        float sum = 0.0;
        for (int pos_index = 0; pos_index < pos + 1; pos_index++) {
            kq_m[i][pos_index] = 0.0;
            for (int j = 0; j < kqv_dim; j++) {
                kq_m[i][pos_index] += q_m[i][j] * k_m[pos_index][i][j];
            }
            kq_m[i][pos_index] = expf(kq_m[i][pos_index] / sqrt_d);
            sum += kq_m[i][pos_index];
        }
        for (int pos_index = 0; pos_index < pos + 1; pos_index++) {
            kq_m[i][pos_index] /= sum;
        }
    }
}

// TODO: current implementation is does not support GQA
void scale_v_mul(float *kq, float *v_cache, float *z, int n_heads, int n_kv_heads, int pos, int kqv_dim) {
    // q heads per kv head
    // int kv_head_ratio = n_heads / n_kv_heads;

    using kq_matrix_t = float(&)[n_heads][pos + 1];
    using v_matrix_t = float(&)[pos + 1][n_heads][kqv_dim];
    using z_matrix_t = float(&)[n_heads][kqv_dim];

    kq_matrix_t kq_m = reinterpret_cast<kq_matrix_t>(*kq);
    v_matrix_t v_m = reinterpret_cast<v_matrix_t>(*v_cache);
    z_matrix_t z_m = reinterpret_cast<z_matrix_t>(*z);

    // TODO: add simd
    // this can be potentially blocked for better cache performance
    // (not sure if rows in z will be cached)
#pragma omp parallel for
    for (int i = 0; i < n_heads; i++) {
        for (int j = 0; j < kqv_dim; j++) {
            z_m[i][j] = 0.0f;
        }

        for (int pos_index = 0; pos_index < pos + 1; pos_index++) {
            for (int j = 0; j < kqv_dim; j++) {
                z_m[i][j] += v_m[pos_index][i][j] * kq_m[i][pos_index];
            }
        }
    }
}

void ffn(float *w1, float *w2, float *w3, float *x, float *x2, float *x_res, int hidden_dim, int embed_dim) {
    b_matvec(w1, x, x2, hidden_dim, embed_dim);
    // swish can be fused with matvecmul (actually maybe all can be fused)
    inplace_swish(x2, hidden_dim);
    b_matvecmul(w3, x, x2, hidden_dim, embed_dim);
    b_matvecadd(w2, x2, x_res, embed_dim, hidden_dim);
}

float *forward(Transformer *transformer, int token, int pos) {
    // TODO: change this lol
    State *state = transformer->state;
    Weights *weights = transformer->weights;
    Config *config = transformer->config;

    float *x_res = state->x_res;
    float *x = state->x;
    float *x2 = state->x2;
    float *k_cache = state->k_cache;
    float *v_cache = state->v_cache;
    float *q = state->q;
    float *kq = state->kq;
    float *z = state->z;
    float *logits = state->logits;

    float *wq = weights->wq;
    float *wk = weights->wk;
    float *wv = weights->wv;
    float *wo = weights->wo;
    float *rms_att_w = weights->rms_att_w;
    float *rms_ffn_w = weights->rms_ffn_w;
    float *rms_final_w = weights->rms_final_w;
    float *w1 = weights->w1;
    float *w2 = weights->w2;
    float *w3 = weights->w3;
    float *wcls = weights->wcls;
    float *token_embedding_table = weights->token_embedding_table;

    int n_layers = config->n_layers;
    int n_heads = config->n_heads;
    int n_kv_heads = config->n_kv_heads;
    int embed_dim = config->embed_dim;
    int hidden_dim = config->hidden_dim;
    int kqv_dim = embed_dim / n_heads;
    int vocab_size = config->vocab_size;

    // copy the token embedding into x
    float *content_row = token_embedding_table + token * embed_dim;
    memcpy(x_res, content_row, embed_dim * sizeof(float));
    // printf("pos: %d\n\n\n", pos);

    for (int layer = 0; layer < n_layers; layer++) {
        // print(rms_att_w, 1, embed_dim);
        // print(x, 1, embed_dim);
        rms_norm(x_res, rms_att_w, x, embed_dim);

        // print(x, 1, embed_dim);

        // calculate new q, k, v and store k, v in cache
        b_matvec(wq, x, q, kqv_dim * n_heads, embed_dim);
        b_matvec(wk, x, k_cache + pos * kqv_dim * n_kv_heads, kqv_dim * n_kv_heads, embed_dim);
        b_matvec(wv, x, v_cache + pos * kqv_dim * n_kv_heads, kqv_dim * n_kv_heads, embed_dim);

        // if (pos == 1) {
        //     print(q, 1, kqv_dim * n_heads);
        //     print(k_cache + pos * kqv_dim * n_kv_heads, 1, kqv_dim * n_kv_heads);
        //     print(v_cache + pos * kqv_dim * n_kv_heads, 1, kqv_dim * n_kv_heads);
        //     exit(0);
        // }

        // rope
        for (int i = 0; i < n_heads; i++)
            inplace_rope(q + kqv_dim * i, pos, kqv_dim);
        for (int i = 0; i < n_kv_heads; i++)
            inplace_rope(k_cache + pos * kqv_dim * n_kv_heads + kqv_dim * i, pos, kqv_dim);

        // if (pos == 1) {
        //     print(q, 1, kqv_dim * n_heads);
        //     print(k_cache + pos * kqv_dim * n_kv_heads, 1, kqv_dim * n_kv_heads);
        // exit(0);
        // }

        // calculate multihead scale for v_cache
        kq_mul_softmax_fused(q, k_cache, kq, n_heads, n_kv_heads, pos, kqv_dim);

        // if (pos == 1)
        //     print(kq, n_heads, pos + 1);

        // calculate z
        scale_v_mul(kq, v_cache, z, n_heads, n_kv_heads, pos, kqv_dim);

        // if (pos == 1)
        //     print(z, 1, n_heads * kqv_dim);
        // exit(0);

        // multiply z, wo and add residual
        b_matvecadd(wo, z, x_res, embed_dim, n_heads * kqv_dim);

        // if (pos == 1) {
        //     print(x_res, 1, embed_dim);
        //     exit(0);
        // }

        // ffn
        rms_norm(x_res, rms_ffn_w, x, embed_dim);
        ffn(w1, w2, w3, x, x2, x_res, hidden_dim, embed_dim);

        // print(x_res, 1, embed_dim);
        // exit(0);

        wq += kqv_dim * n_heads * embed_dim;
        wk += kqv_dim * n_kv_heads * embed_dim;
        wv += kqv_dim * n_kv_heads * embed_dim;
        wo += embed_dim * n_heads * kqv_dim;
        rms_att_w += embed_dim;
        rms_ffn_w += embed_dim;
        w1 += embed_dim * hidden_dim;
        w2 += embed_dim * hidden_dim;
        w3 += embed_dim * hidden_dim;
        int seq_length = 128;  // TODO: fix
        k_cache += kqv_dim * n_kv_heads * seq_length;
        v_cache += kqv_dim * n_kv_heads * seq_length;
    }

    // print(x_res, 1, embed_dim);
    // exit(0);

    rms_norm(x_res, rms_final_w, x_res, embed_dim);

    // print(x_res, 1, embed_dim);
    // exit(0);

    // classifier into logits
    b_matvec(wcls, x_res, logits, vocab_size, embed_dim);

    // print(logits, 1, 100);
    // exit(0);

    return logits;
}

void alloc_state(Transformer *transformer, size_t alignment, int cache_size) {
    State *state = transformer->state;
    Weights *weights = transformer->weights;
    Config *config = transformer->config;

    int n_layers = config->n_heads;
    int n_heads = config->n_heads;
    int n_kv_heads = config->n_kv_heads;
    int embed_dim = config->embed_dim;
    int hidden_dim = config->hidden_dim;
    int kqv_dim = embed_dim / n_heads;
    int vocab_size = config->vocab_size;

    state->x_res = (float *)aligned_alloc(alignment, sizeof(float) * embed_dim);
    state->x = (float *)aligned_alloc(alignment, sizeof(float) * embed_dim);
    state->x2 = (float *)aligned_alloc(alignment, sizeof(float) * embed_dim);
    state->k_cache = (float *)aligned_alloc(alignment, sizeof(float) * n_layers * n_kv_heads * kqv_dim * cache_size);
    state->v_cache = (float *)aligned_alloc(alignment, sizeof(float) * n_layers * n_kv_heads * kqv_dim * cache_size);
    state->q = (float *)aligned_alloc(alignment, sizeof(float) * n_heads * kqv_dim);
    state->kq = (float *)aligned_alloc(alignment, sizeof(float) * n_heads * cache_size);
    state->z = (float *)aligned_alloc(alignment, sizeof(float) * n_heads * kqv_dim);
    state->logits = (float *)aligned_alloc(alignment, sizeof(float) * vocab_size);

    if (state->x_res == NULL || state->x == NULL || state->x2 == NULL || state->k_cache == NULL ||
        state->v_cache == NULL || state->q == NULL || state->kq == NULL || state->z == NULL ||
        state->logits == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
}

void rearrange_matrix(float *w, int d, int n, int block_size) {
    float *copy = (float *)malloc(sizeof(float) * d * n);
    if (copy == NULL) {
        printf("error malloc failed\n");
        exit(1);
    }

    using blocked_matrix_t = float(&)[d / block_size][n / block_size][block_size][block_size];
    blocked_matrix_t w_m = reinterpret_cast<blocked_matrix_t>(*w);

    for (int i = 0; i < d; i++)
        for (int j = 0; j < n; j++)
            copy[i * n + j] = w[i * n + j];

    for (int i = 0; i < d / block_size; i++)
        for (int ii = 0; ii < block_size; ii++)
            for (int j = 0; j < n / block_size; j++)
                for (int jj = 0; jj < block_size; jj++)
                    w_m[i][j][jj][ii] = copy[i * block_size * n + ii * n + j * block_size + jj];

    free(copy);
}

void rearrange_weights(float *w, int d, int n, int n_layers) {
    for (int i = 0; i < n_layers; i++)
        rearrange_matrix(w + i * d * n, d, n, 4);
}

void memory_map_weights(Weights *weights, Config *config, float *ptr, int shared_weights) {
    int head_size = config->embed_dim / config->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = config->n_layers;
    weights->token_embedding_table = ptr;
    ptr += config->vocab_size * config->embed_dim;
    weights->rms_att_w = ptr;
    ptr += n_layers * config->embed_dim;
    weights->wq = ptr;
    ptr += n_layers * config->embed_dim * (config->n_heads * head_size);
    weights->wk = ptr;
    ptr += n_layers * config->embed_dim * (config->n_kv_heads * head_size);
    weights->wv = ptr;
    ptr += n_layers * config->embed_dim * (config->n_kv_heads * head_size);
    weights->wo = ptr;
    ptr += n_layers * (config->n_heads * head_size) * config->embed_dim;
    weights->rms_ffn_w = ptr;
    ptr += n_layers * config->embed_dim;
    weights->w1 = ptr;
    ptr += n_layers * config->embed_dim * config->hidden_dim;
    weights->w2 = ptr;
    ptr += n_layers * config->hidden_dim * config->embed_dim;
    weights->w3 = ptr;
    ptr += n_layers * config->embed_dim * config->hidden_dim;
    weights->rms_final_w = ptr;
    ptr += config->embed_dim;
    ptr += config->seq_len * head_size / 2;  // skip what used to be freq_cis_real (for RoPE)
    ptr += config->seq_len * head_size / 2;  // skip what used to be freq_cis_imag (for RoPE)

    if (shared_weights) {
        weights->wcls = (float *)aligned_alloc(128, sizeof(float) * config->embed_dim * config->vocab_size);
        for (int i = 0; i < config->vocab_size * config->embed_dim; i++) {
            weights->wcls[i] = weights->token_embedding_table[i];
        }
    } else {
        weights->wcls = ptr;
    }

    rearrange_weights(weights->wq, config->n_heads * head_size, config->embed_dim, (int)n_layers);
    rearrange_weights(weights->wk, config->n_kv_heads * head_size, config->embed_dim, (int)n_layers);
    rearrange_weights(weights->wv, config->n_kv_heads * head_size, config->embed_dim, (int)n_layers);
    rearrange_weights(weights->wo, config->embed_dim, config->n_heads * head_size, (int)n_layers);
    rearrange_weights(weights->w1, config->hidden_dim, config->embed_dim, (int)n_layers);
    rearrange_weights(weights->w2, config->embed_dim, config->hidden_dim, (int)n_layers);
    rearrange_weights(weights->w3, config->hidden_dim, config->embed_dim, (int)n_layers);
    rearrange_weights(weights->wcls, config->vocab_size, config->embed_dim, 1);
}

void read_checkpoint(char *checkpoint, Config *config, Weights *weights,
                     int *fd, float **data, ssize_t *file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END);  // move file pointer to end of file
    *file_size = ftell(file);  // get the file size, in bytes
    fclose(file);

    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY);  // open in read only mode
    if (*fd == -1) {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = (float *)mmap(NULL, *file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    float *weights_ptr = *data + sizeof(Config) / sizeof(float);

    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void free_state(Transformer *transformer) {
    State *state = transformer->state;

    free(state->x_res);
    free(state->x);
    free(state->x2);
    free(state->k_cache);
    free(state->v_cache);
    free(state->q);
    free(state->kq);
    free(state->z);
    free(state->logits);

    state->x_res = NULL;
    state->x = NULL;
    state->x2 = NULL;
    state->k_cache = NULL;
    state->v_cache = NULL;
    state->q = NULL;
    state->kq = NULL;
    state->z = NULL;
    state->logits = NULL;
}

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];  // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;  // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';  // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) {
        return;
    }
    if (piece[0] == '\0') {
        return;
    }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;  // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = {.str = str};  // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {
        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c;  // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;  // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;  // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;  // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

int sample_argmax(float *probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(Transformer *transformer, Tokenizer *tokenizer, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) {
        prompt = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));  // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;                // used to time our code, only initialized after first iteration
    int next;                      // will store the next token in the sequence
    int token = prompt_tokens[0];  // kick off with the first token in the prompt
    int pos = 0;                   // position in the sequence
    int vocab_size = transformer->config->vocab_size;
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);
        // printf(" [%d] ", token);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample_argmax(logits, vocab_size);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) {
            break;
        }

        // print(logits, 1, 100);

        // print the token as string, decode it with the Tokenizer object
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece);  // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) {
            start = time_in_ms();
        }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }

    free(prompt_tokens);
}

int main(int argc, char *argv[]) {
    // Timer timer;

    // omp_set_num_threads(32);

    // timer.start();
    Config config;
    Weights weights;
    State state;
    Transformer transformer;

    transformer.config = &config;
    transformer.state = &state;
    transformer.weights = &weights;

    ssize_t size;
    int fd;
    float *data;

    char *model_file = "stories110M.bin";

    read_checkpoint(model_file, &config, &weights, &fd, &data, &size);

    char *tokenizer_file = "tokenizer.bin";
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_file, transformer.config->vocab_size);

    alloc_state(&transformer, 128, 128);

    // float *output = forward(&transformer, 1, 0);
    // print(output, 1, 100);
    // output = forward(&transformer, 1, 0);

    generate(&transformer, &tokenizer, "Once upon a time", 100);

    return 0;
}