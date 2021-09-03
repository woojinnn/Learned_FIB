#ifndef LEARNED_FIB_H
#define LEARNED_FIB_H

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "NN.h"
#include "PWL.h"

#define CALCULATE_PREFIX(x) (x >> (32 - 8))

template <typename KeyType>
class Learned_FIB {
   private:
    std::array<std::vector<KeyType>, (size_t)pow(2, 8)> splitted_dataset;

    std::array<std::vector<std::pair<KeyType, uint64_t>>, (size_t)pow(2, 8)> boundaries;
    std::vector<uint64_t> starting_points;

    std::vector<std::unique_ptr<NN<KeyType>>> NeuralNetworks;

    void load_dataset(const std::string& path);
    void derive_boundaries(uint32_t model_idx);
    uint64_t error_threshold;

   public:
    void train(const std::string& path, double threshold);
    uint64_t get_error_threshold() { return error_threshold; }
    uint64_t find(KeyType key);

    void load(const std::string& path);
    void save(const std::string& path);
};

template <typename KeyType>
void Learned_FIB<KeyType>::load_dataset(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        std::cerr << "[Learned_FIB.h] load dataset failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    // read data size
    uint64_t dataset_size;
    is.read(reinterpret_cast<char*>(&dataset_size), sizeof(uint64_t));

    KeyType prev_prefix = 999;
    KeyType key;
    for (uint64_t i = 0; i < dataset_size; ++i) {
        is.read(reinterpret_cast<char*>(&key), sizeof(KeyType));
        if (CALCULATE_PREFIX(key) != prev_prefix) {
            starting_points.push_back(i);
            prev_prefix = CALCULATE_PREFIX(key);
        }
        splitted_dataset[CALCULATE_PREFIX(key)].push_back(key);
    }
    is.close();
}

template <typename KeyType>
void Learned_FIB<KeyType>::derive_boundaries(uint32_t model_idx) {
    double a, b;    // variable for slope and bias
    double p, err;  // variable for error calculation

    // B <- {(x_0, y_0)}
    boundaries[model_idx].push_back(std::make_pair(splitted_dataset[model_idx][0], starting_points[model_idx]));

    // l, r: left and right boundary of a line segment
    uint64_t l = 0, r = 2;
    while (r < splitted_dataset[model_idx].size()) {
        if (splitted_dataset[model_idx][r] == splitted_dataset[model_idx][l]) {  // for duplicate keys
            while (splitted_dataset[model_idx][r] == splitted_dataset[model_idx][l]) {
                r++;
            }
            r++;
            continue;
        }

        // Derive a line's (slope, bias) passing through (x_l, l) and (x_r, r)
        a = static_cast<double>(r - l) / static_cast<double>(splitted_dataset[model_idx][r] - splitted_dataset[model_idx][l]);
        b = l - a * static_cast<double>(splitted_dataset[model_idx][l]);

        // Examine the error between x_(l+1) and x_(r-1)
        for (uint64_t i = l + 1; i < r; ++i) {
            p = a * static_cast<double>(splitted_dataset[model_idx][i]) + b;  // compute the y-value on the line for the x-value of x_i
            double err = p > static_cast<double>(i) ? p - static_cast<double>(i) : static_cast<double>(i) - p;
            if (err >= error_threshold) {
                // The error is geq than the error_threshold
                // Append (x_(r-1), r-1) to B
                boundaries[model_idx].push_back(std::make_pair(splitted_dataset[model_idx][r - 1], r - 1 + starting_points[model_idx]));

                l = r - 1;
                break;
            }
        }
        r++;
    }

    // insert last point if not inserted
    if (splitted_dataset[model_idx][splitted_dataset[model_idx].size() - 1] != boundaries[model_idx][boundaries[model_idx].size() - 1].first)
        boundaries[model_idx].push_back(std::make_pair(splitted_dataset[model_idx][splitted_dataset[model_idx].size() - 1], splitted_dataset[model_idx].size() - 1 + starting_points[model_idx]));  // last point
}

template <typename KeyType>
void Learned_FIB<KeyType>::train(const std::string& path, double threshold) {
    // load dataset
    load_dataset(path);

    // 1 for double-uint64_t rounding-up issue
    error_threshold = static_cast<uint64_t>(std::abs(threshold));

    for (uint32_t model_idx = 0; model_idx < pow(2, 8); ++model_idx) {
        // make PWL function
        derive_boundaries(model_idx);

        std::unique_ptr<NN<KeyType>> nn(new NN<KeyType>());
        NeuralNetworks.push_back(std::move(nn));

        // train NN using boundaries
        NeuralNetworks[model_idx]->train(boundaries[model_idx].begin(), boundaries[model_idx].end());
    }
}

template <typename KeyType>
uint64_t Learned_FIB<KeyType>::find(KeyType key) {
    return static_cast<uint64_t>(NeuralNetworks[CALCULATE_PREFIX(key)]->inference(key));
}

template <typename KeyType>
void Learned_FIB<KeyType>::save(const std::string& path) {
    for (uint32_t i = 0; i < pow(2, 8); ++i) {
        NeuralNetworks[i]->save(path + "_" + std::to_string(i));
    }
}

template <typename KeyType>
void Learned_FIB<KeyType>::load(const std::string& path) {
    for (uint32_t i = 0; i < pow(2, 8); ++i) {
        std::unique_ptr<NN<KeyType>> nn(new NN<KeyType>());
        NeuralNetworks.push_back(std::move(nn));
        NeuralNetworks[i]->load(path + "_" + std::to_string(i));
    }
}

#endif  // LEARNED_FIB_H