#include <arm_neon.h>
#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "params.h"
#include "timer.h"
#include "utilities.h"

using simd_matrix_t = float16_t (&)[MATRIX_SIZE / 8][MATRIX_SIZE / 8][8][8];

void matmul_simd2(simd_matrix_t w, float16_t *x, float16_t *x_out, int d, int n) {
    int BLOCKSIZE = 8;
#pragma omp parallel for
    for (int i = 0; i < d / BLOCKSIZE; i++) {
        float16x8_t v_out = vdupq_n_f32(0.0);
        for (int j = 0; j < n / BLOCKSIZE; j++) {
            for (int ii = 0; ii < BLOCKSIZE; ii++) {
                float16x8_t v1 = vld1q_f16((float16_t *)&w[i][j][ii]);
                float16x8_t v2 = vdupq_n_f16(x[i * BLOCKSIZE + ii]);
                v_out = vfmaq_f16(v_out, v1, v2);
            }
        }
        vst1q_f16(&x_out[i * BLOCKSIZE], v_out);
    }
}

void matmul_simd(float16_t *w, float16_t *x, float16_t *x_out, int d, int n) {
    int BLOCKSIZE = 8;
#pragma omp parallel for
    for (int j = 0; j < d / BLOCKSIZE; j++) {
        float16x8_t v_out = vdupq_n_f32(0.0);

        for (int i = 0; i < n / BLOCKSIZE; i++) {
            // float32x4_t v_out = vld1q_f32(&x_out[j * BLOCKSIZE]);

            for (int ii = 0; ii < BLOCKSIZE; ii++) {
                float16x8_t v1 = vld1q_f16(&w[i * d * BLOCKSIZE + ii * d + j * BLOCKSIZE]);
                float16x8_t v2 = vdupq_n_f16(x[i * BLOCKSIZE + ii]);
                v_out = vfmaq_f16(v_out, v1, v2);
            }

            // vst1q_f32(&x_out[j * BLOCKSIZE], v_out);
        }

        vst1q_f16(&x_out[j * BLOCKSIZE], v_out);
    }
}

void matmul_improved(float16_t *w, float16_t *x, float16_t *x_out, int d, int n) {
    int BLOCKSIZE = 8;
#pragma omp parallel for
    for (int i = 0; i < d / BLOCKSIZE; i++)
        for (int j = 0; j < n / BLOCKSIZE; j++)
            for (int ii = 0; ii < BLOCKSIZE; ii++)
                for (int jj = 0; jj < BLOCKSIZE; jj++)
                    x_out[i * BLOCKSIZE + ii] += w[i * n * BLOCKSIZE + ii * n + j * BLOCKSIZE + jj] * x[j * BLOCKSIZE + jj];
}

// w (d, n) @ x (n) = xout (d)
void matmul_truth(float16_t *w, float16_t *x, float16_t *x_out, int d, int n) {
#pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float16_t val = 0;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        x_out[i] = val;
    }
}

int main(int argc, char *argv[]) {
    Timer timer;

    omp_set_num_threads(32);

    int d = MATRIX_SIZE;
    int n = MATRIX_SIZE;

    double blocked_time;
    double simd_time;
    double baseline_time;

    float16_t *w_m = (float16_t *)aligned_alloc(ALIGNMENT, sizeof(float16_t) * d * n);
    float16_t *x = (float16_t *)aligned_alloc(ALIGNMENT, sizeof(float16_t) * n);
    float16_t *x_out = (float16_t *)aligned_alloc(ALIGNMENT, sizeof(float16_t) * d);
    float16_t *x_out_truth = (float16_t *)aligned_alloc(ALIGNMENT, sizeof(float16_t) * d);

    init_matricies(w_m, d * n);
    init_matricies(x, n);

    // rearrange w1
    float16_t *w_mT = (float16_t *)aligned_alloc(ALIGNMENT, sizeof(float16_t) * n * d);
    simd_matrix_t s_w = reinterpret_cast<simd_matrix_t>(*w_mT);
    int blocksize = 8;
#pragma omp parallel
    for (int i = 0; i < d / blocksize; i++)
        for (int ii = 0; ii < blocksize; ii++)
            for (int j = 0; j < n / blocksize; j++)
                for (int jj = 0; j < blocksize; j++)
                    s_w[j][i][jj][ii] = w_m[i * blocksize * n + ii * n + j * blocksize + jj];

    //     float16_t *w_mT = (float16_t *)aligned_alloc(ALIGNMENT, sizeof(float16_t) * n * d);
    // #pragma omp parallel
    //     for (int i = 0; i < d; i++)
    //         for (int j = 0; j < n; j++)
    //             w_mT[j * d + i] = w_m1[i * n + j];

    for (int i = 0; i < 10; i++) {
        // clear x_out
#pragma omp parallel
        for (int i = 0; i < d; i++)
            x_out[i] = 0.0;

        timer.start();
        // matmul_improved(w_m, x, x_out, d, n);
        matmul_simd2(s_w, x, x_out, d, n);
        simd_time += timer.stop("simd: ");

        // matmul_truth(w_m, x, x_out_truth, d, n);

//         float diff;
// #pragma omp parallel for reduction(max : diff)
//         for (int i = 0; i < d; i++)
//             diff = std::max(diff, std::abs(x_out_truth[i] - x_out[i]));

//         std::cout << "max diff: " << diff << std::endl;
    }

    std::cout << std::endl;
    std::cout << "simd total: " << simd_time << std::endl;
    // std::cout << "blocked total: " << blocked_time << std::endl;
    // std::cout << "baseline total: " << baseline_time << std::endl;

    return 0;
}