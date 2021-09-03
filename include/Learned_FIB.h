#ifndef LEARNED_FIB_H
#define LEARNED_FIB_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "NN.h"
#include "PWL.h"

#define CALCULATE_PREFIX(x) (x >> (32 - 4))

template <typename KeyType>
class Learned_FIB {
   private:
    std::vector<KeyType> dataset;
    std::vector<std::pair<KeyType, uint64_t>> boundaries;
    std::vector<uint64_t> branching_points;
    std::vector<std::unique_ptr<NN<KeyType>>> NeuralNetworks;

    void load_dataset(const std::string& path);
    void derive_boundaries();
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
    dataset.resize(dataset_size);

    // Read keys
    is.read(reinterpret_cast<char*>(dataset.data()), dataset_size * sizeof(KeyType));
    is.close();
}

template <typename KeyType>
void Learned_FIB<KeyType>::derive_boundaries() {
    double a, b;    // variable for slope and bias
    double p, err;  // variable for error calculation

    // B <- {(x_0, 0)}
    boundaries.push_back(std::make_pair(dataset[0], 0));
    branching_points.push_back(static_cast<uint64_t>(0));

    // l, r: left and right boundary of a line segment
    uint64_t l = 0, r = 2;
    while (r < dataset.size()) {
        if (dataset[r] == dataset[l]) {  // for duplicate keys
            while (dataset[r] == dataset[l]) {
                r++;
            }
            r++;
            continue;
        }

        // when datum's prefix becomes different, append it to boundaries and branching_points
        if (CALCULATE_PREFIX(dataset[r]) != CALCULATE_PREFIX(dataset[l])) {
            boundaries.push_back(std::make_pair(dataset[r - 1], r - 1));
            branching_points.push_back(boundaries.size() - 1);
            l = r;
            r += 2;
            continue;
        }

        // Derive a line's (slope, bias) passing through (x_l, l) and (x_r, r)
        a = static_cast<double>(r - l) / static_cast<double>(dataset[r] - dataset[l]);
        b = l - a * static_cast<double>(dataset[l]);

        // Examine the error between x_(l+1) and x_(r-1)
        for (uint64_t i = l + 1; i < r; ++i) {
            p = a * static_cast<double>(dataset[i]) + b;  // compute the y-value on the line for the x-value of x_i
            double err = p > static_cast<double>(i) ? p - static_cast<double>(i) : static_cast<double>(i) - p;
            if (err >= error_threshold) {
                // The error is geq than the error_threshold
                // Append (x_(r-1), r-1) to B
                boundaries.push_back(std::make_pair(dataset[r - 1], r - 1));

                l = r - 1;
                break;
            }
        }
        r++;
    }

    if (dataset[dataset.size() - 1] != boundaries[boundaries.size() - 1].first)
        boundaries.push_back(std::make_pair(dataset[dataset.size() - 1], dataset.size() - 1));  // last point

    branching_points.push_back(boundaries.size() - 1);
}

template <typename KeyType>
void Learned_FIB<KeyType>::train(const std::string& path, double threshold) {
    // load dataset
    load_dataset(path);

    // 1 for double-uint64_t rounding-up issue
    error_threshold = static_cast<uint64_t>(std::abs(threshold));

    // make PWL function
    derive_boundaries();

    for (uint32_t i = 0; i < pow(2, 4); ++i) {
        std::unique_ptr<NN<KeyType>> nn(new NN<KeyType>());
        NeuralNetworks.push_back(std::move(nn));
    }

    // train NN using boundaries
    for (uint32_t i = 0; i < branching_points.size() - 1; ++i) {
        NeuralNetworks[i]->train(boundaries.begin() + branching_points[i], boundaries.begin() + branching_points[i + 1]);
    }
}

template <typename KeyType>
uint64_t Learned_FIB<KeyType>::find(KeyType key) {
    return static_cast<uint64_t>(NeuralNetworks[CALCULATE_PREFIX(key)]->inference(key));
}

template <typename KeyType>
void Learned_FIB<KeyType>::save(const std::string& path) {
    for (uint32_t i = 0; i < pow(2, 4); ++i) {
        NeuralNetworks[i]->save(path + "_" + std::to_string(i));
    }
}

template <typename KeyType>
void Learned_FIB<KeyType>::load(const std::string& path) {
    for (uint32_t i = 0; i < pow(2, 4); ++i) {
        std::unique_ptr<NN<KeyType>> nn(new NN<KeyType>());
        NeuralNetworks.push_back(std::move(nn));
        NeuralNetworks[i]->load(path + "_" + std::to_string(i));
    }
}

#endif  // LEARNED_FIB_H