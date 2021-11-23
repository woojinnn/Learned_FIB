#ifndef NN_H
#define NN_H

#include <immintrin.h>  // for AVX
#include <stdlib.h>     // aligned_alloc

#include <cmath>  // for absolute value
#include <fstream>
#include <iostream>
#include <utility>  // pair
#include <vector>

template <typename KeyType>
class NN {
   private:
    double *weights1;
    double *weights2;
    double *biases1;
    double bias2;

    uint32_t numNeurons;

   public:
    NN(){};
    void train(const typename std::vector<std::pair<KeyType, uint64_t>>::iterator boundaries_begin,
               const typename std::vector<std::pair<KeyType, uint64_t>>::iterator boundaries_end);
    double inference(KeyType key);

    void load(const std::string &path);
    void save(const std::string &path);
    ~NN();
};

template <typename KeyType>
void NN<KeyType>::train(const typename std::vector<std::pair<KeyType, uint64_t>>::iterator boundaries_begin,
                        const typename std::vector<std::pair<KeyType, uint64_t>>::iterator boundaries_end) {
    numNeurons = boundaries_end - boundaries_begin - 1;  // numNeurons == number of line segments
    weights1 = (double *)aligned_alloc(32, numNeurons * sizeof(double));
    weights2 = (double *)aligned_alloc(32, numNeurons * sizeof(double));
    biases1 = (double *)aligned_alloc(32, numNeurons * sizeof(double));

    bias2 = (*boundaries_begin).second;

    auto tmp = boundaries_begin;
    double prev_slope = 0, cur_slope;
    for (uint32_t i = 0; i < numNeurons; ++i, ++tmp) {
        cur_slope = static_cast<double>((*(tmp + 1)).second - (*tmp).second) / static_cast<double>((*(tmp + 1)).first - (*tmp).first);
        weights1[i] = std::abs(cur_slope - prev_slope);
        biases1[i] = -(weights1[i] * (*tmp).first);
        weights2[i] = cur_slope > prev_slope ? 1 : -1;
        prev_slope = cur_slope;
    }
}

template <typename KeyType>
double NN<KeyType>::inference(KeyType key) {
    double d_key = static_cast<double>(key);
    __m256d vec_input = _mm256_set1_pd(d_key);
    __m256d vec_zeros = _mm256_setzero_pd();
    __m256d sum_vec = _mm256_setzero_pd();

    double result;
    for (uint32_t i = 0; i < numNeurons / 4; ++i) {
        __m256d vec_weights1 = _mm256_load_pd(weights1 + i * 4);
        __m256d vec_biases1 = _mm256_load_pd(biases1 + i * 4);
        __m256d vec_weights2 = _mm256_load_pd(weights2 + i * 4);

        __m256d layer1 = _mm256_fmadd_pd(vec_weights1, vec_input, vec_biases1);
        layer1 = _mm256_max_pd(layer1, vec_zeros);  // RELU

        __m256d tmp = _mm256_mul_pd(layer1, vec_weights2);
        sum_vec = _mm256_add_pd(sum_vec, tmp);
    }

    __m256d temp = _mm256_hadd_pd(sum_vec, sum_vec);
    __m128d sum_high = _mm256_extractf128_pd(temp, 1);
    __m128d sum_all = _mm_add_pd(sum_high, _mm256_castpd256_pd128(temp));
    result = ((double *)&sum_all)[0];

    for (uint32_t i = numNeurons - numNeurons % 4; i < numNeurons; ++i) {
        double tmp = weights1[i] * d_key + biases1[i];
        tmp = std::max(tmp, static_cast<double>(0));
        result += weights2[i] * tmp;
    }

    return result + bias2;
}

template <typename KeyType>
void NN<KeyType>::load(const std::string &path) {
    std::ifstream is(path, std::ios::binary);
    if (is.is_open()) {
        is.read(reinterpret_cast<char *>(&numNeurons), sizeof(uint32_t));

        weights1 = (double *)aligned_alloc(32, numNeurons * sizeof(double));
        weights2 = (double *)aligned_alloc(32, numNeurons * sizeof(double));
        biases1 = (double *)aligned_alloc(32, numNeurons * sizeof(double));

        is.read(reinterpret_cast<char *>(weights1), numNeurons * sizeof(double));
        is.read(reinterpret_cast<char *>(biases1), numNeurons * sizeof(double));
        is.read(reinterpret_cast<char *>(weights2), numNeurons * sizeof(double));

        is.read(reinterpret_cast<char *>(&bias2), sizeof(double));
        is.close();
    } else {
        std::cerr << "[NN.h save] Cannot open " + path << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename KeyType>
void NN<KeyType>::save(const std::string &path) {
    std::ofstream os(path, std::ios::binary);
    if (os.is_open()) {
        os.write(reinterpret_cast<char *>(&numNeurons), sizeof(uint32_t));

        os.write(reinterpret_cast<char *>(weights1), numNeurons * sizeof(double));
        os.write(reinterpret_cast<char *>(biases1), numNeurons * sizeof(double));
        os.write(reinterpret_cast<char *>(weights2), numNeurons * sizeof(double));

        os.write(reinterpret_cast<char *>(&bias2), sizeof(double));
        os.close();
    } else {
        std::cerr << "[NN.h save] Cannot open " + path << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename KeyType>
NN<KeyType>::~NN() {
    free(weights1);
    free(weights2);
    free(biases1);
}

#endif  // NN_H