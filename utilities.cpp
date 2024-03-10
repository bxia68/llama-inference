#include <arm_neon.h>

#include <memory>
#include <new>
#include <random>

#include "params.h"

void init_matricies(float16_t* A, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float16_t> uniform_dist(-1., 1.);
    for (int i = 0; i < size; i++)
        A[i] = uniform_dist(gen);
}