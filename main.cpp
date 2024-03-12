// #include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "params.h"
#include "timer.h"
#include "utilities.h"

// TODO: make a struct to organize these
float *x;

float *k_cache;
float *v_cache;

int n_heads;
int n_kv_heads;
int kqv_dim;
int embed_dim;

int pos;

float *q_w;
float *k_w;
float *v_w;
float *wo;

float *q;
float *kq;
float *z;

void print(float16_t *x, int d, int n) {
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << x[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

// TODO: simd :)
void inplace_swish(float *x, float b, int d) {
#pragma omp parallel for
    for (int i = 0; i < d; i++)
        x[i] = x[i] / (1 + expf(-x[i] * b));
}

// TODO: simd :)
void inplace_rms_norm(float *x, float *w, int d) {
    float rms = 0.0;
#pragma omp parallel for reduction(+ : rms)
    for (int i = 0; i < d; i++)
        rms += x[i] * x[i];

    rms = sqrt(rms / d);

#pragma omp parallel for
    for (int i = 0; i < d; i++)
        x[i] *= w[i] / rms;
}

void inplace_softmax(float *x, int d) {
    float sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < d; i++) {
        x[i] = expf(x[i]);
        sum += x[i];
    }

#pragma omp parallel for
    for (int i = 0; i < d; i++) {
        x[i] /= sum;
    }
}

// w (d, n) @ x (n) = xout (d)
void b_matvec(float *w, float *x, float *x_out, int d, int n) {
    int BLOCKSIZE = 4;
#pragma omp parallel for
    for (int i = 0; i < d / BLOCKSIZE; i++) {
        float32x4_t v_out = vdupq_n_f32(0.);
        for (int j = 0; j < n / BLOCKSIZE; j++) {
            for (int ii = 0; ii < BLOCKSIZE; ii++) {
                float32x4_t v1 = vld1q_f32((float *)&w[i * n * BLOCKSIZE * BLOCKSIZE + j * BLOCKSIZE * BLOCKSIZE + ii * BLOCKSIZE]);
                float32x4_t v2 = vdupq_n_f32(x[j * BLOCKSIZE + ii]);
                v_out = vfmaq_f32(v_out, v1, v2);
            }
        }
        vst1q_f32(&x_out[i * BLOCKSIZE], v_out);
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

void inplace_rope(float *x, int pos, int dim) {
    // TODO: maybe can precalculate power (theta)
    for (int i = 0; i < dim; i += 2) {
        // is there supposed to be a -2 in the exponent?
        float theta = powf(10000.0, -2 * i / (float)dim);
        float val = pos * theta;
        x[i] = x[i] * cosf(val) - x[i + 1] * sinf(val);
        x[i + 1] = x[i + 1] * cosf(val) - x[i] * sinf(val);
    }
}

// TODO: current implementation is does not support GQA
void kq_mul_softmax_fused(float *q, float *k_cache, float *kq, int n_heads, int n_kv_heads, int pos, int kqv_dim) {
    // q heads per kv head
    // int kv_head_ratio = n_heads / n_kv_heads;

    using q_matrix_t = float(&)[n_heads][kqv_dim];
    using k_matrix_t = float(&)[pos][n_heads][kqv_dim];
    using kq_matrix_t = float(&)[n_heads][pos];

    q_matrix_t q_m = reinterpret_cast<q_matrix_t>(*q);
    k_matrix_t k_m = reinterpret_cast<k_matrix_t>(*k_cache);
    kq_matrix_t kq_m = reinterpret_cast<kq_matrix_t>(*kq);

    float sqrt_d = sqrtf((float)kqv_dim);

    // TODO: add simd
    // no sequential i/o on k_cache but fused operation (need to test performance)
#pragma omp parallel for
    for (int i = 0; i < n_heads; i++) {
        float sum = 0.0;
        for (int pos_index = 0; pos_index < pos; pos_index++) {
            for (int j = 0; j < kqv_dim; j++) {
                kq_m[pos_index][i] += q_m[i][j] * k_m[pos_index][i][j];
            }
            kq_m[i][pos_index] = expf(kq_m[i][pos_index] / sqrt_d);
            sum += kq_m[i][pos_index];
        }
        for (int pos_index = 0; pos_index < pos; pos_index++) {
            kq_m[i][pos_index] /= sum;
        }
    }
}

// TODO: current implementation is does not support GQA
void scale_v_mul(float *kq, float *v_cache, float *z, int n_heads, int n_kv_heads, int pos, int kqv_dim) {
    // q heads per kv head
    // int kv_head_ratio = n_heads / n_kv_heads;

    using kq_matrix_t = float(&)[n_heads][pos];
    using v_matrix_t = float(&)[pos][n_heads][kqv_dim];
    using z_matrix_t = float(&)[n_heads][kqv_dim];

    kq_matrix_t kq_m = reinterpret_cast<kq_matrix_t>(*kq);
    v_matrix_t v_m = reinterpret_cast<v_matrix_t>(*v_cache);
    z_matrix_t z_m = reinterpret_cast<z_matrix_t>(*z);

    float sqrt_d = sqrtf((float)kqv_dim);

    // TODO: add simd
    // this can be potentially blocked for better cache performance
    // (not sure if rows in z will be cached)
#pragma omp parallel for
    for (int i = 0; i < n_heads; i++) {
        for (int pos_index = 0; pos_index < pos; pos_index++) {
            for (int j = 0; j < kqv_dim; j++) {
                z_m[pos_index][i] += v_m[pos_index][i][j] * kq_m[i][pos_index];
            }
        }
    }
}

void attention() {
    // calculate new q, k, v and store k, v in cache
    b_matvec(q_w, x, q, kqv_dim * n_heads, embed_dim);
    b_matvec(q_w, x, k_cache + pos * kqv_dim * n_kv_heads, kqv_dim * n_kv_heads, embed_dim);
    b_matvec(q_w, x, v_cache + pos * kqv_dim * n_kv_heads, kqv_dim * n_kv_heads, embed_dim);

    // rope
    for (int i = 0; i < n_heads; i++)
        inplace_rope(q + kqv_dim * i, pos, kqv_dim);
    for (int i = 0; i < n_kv_heads; i++)
        inplace_rope(k_cache + pos * kqv_dim * n_kv_heads + kqv_dim * i, pos, kqv_dim);

    // calculate multihead scale for v_cache
    kq_mul_softmax_fused(q, k_cache, kq, n_heads, n_kv_heads, pos, kqv_dim);

    // calculate z and output
    scale_v_mul(kq, v_cache, z, n_heads, n_kv_heads, pos, kqv_dim);

    b_matvec(wo, z, x, embed_dim, n_heads * kqv_dim);
}

int main(int argc, char *argv[]) {
    Timer timer;

    omp_set_num_threads(32);

    // float *w_m = (float *)aligned_alloc(ALIGNMENT, sizeof(float) * d * n);
    // float *x = (float *)aligned_alloc(ALIGNMENT, sizeof(float) * n);
    // float *x_out = (float *)aligned_alloc(ALIGNMENT, sizeof(float) * d);
    // float *x_out_truth = (float *)aligned_alloc(ALIGNMENT, sizeof(float) * d);

    // timer.start();

    return 0;
}