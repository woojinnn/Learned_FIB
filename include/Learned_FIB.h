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

template <typename KeyType>
class Learned_FIB {
   private:
    inline KeyType calculate_prefix(KeyType key) {
        return (key >> ((unsigned int)(sizeof(KeyType)) * 8 - prefix));
    }

    // for training
    std::vector<KeyType> splitted_dataset;
    std::vector<std::pair<KeyType, uint64_t>> boundaries;
    uint64_t starting_point;
    void derive_boundaries(void);
    uint64_t dataset_size;
    uint64_t error_threshold;

    // whole model
    unsigned int prefix;
    std::vector<std::unique_ptr<NN<KeyType>>> NeuralNetworks;

    // validation
    uint64_t model_max_error;  // error boundary

    // for debugging
    void check_max_error(NN<KeyType>* nn);

   public:
    Learned_FIB(unsigned int prefix_len) {
        prefix = prefix_len;
        starting_point = 0;
        model_max_error = 0;
    };
    void train(const std::string& dataset_path, const std::string& model_path, double threshold);
    uint64_t get_max_error(void) { return model_max_error; }
    uint64_t find(KeyType key);  // prediction

    // load saved model
    void load(const std::string& model_path);
};

template <typename KeyType>
void Learned_FIB<KeyType>::check_max_error(NN<KeyType>* nn) {
    uint64_t max_err = 0;
    uint64_t tmp_err;
    uint64_t pred;

    for (uint64_t i = 0; i < splitted_dataset.size(); ++i) {
        pred = (uint64_t)nn->inference(splitted_dataset[i]);
        tmp_err = pred > i + starting_point ? pred - (i + starting_point) : (i + starting_point) - pred;
        if (tmp_err > max_err)
            max_err = tmp_err;
    }

    if (max_err > model_max_error) {
        model_max_error = max_err;
    }

    std::cout << "data size: " << splitted_dataset.size() << "\t, max err: " << max_err << std::endl;
}

template <typename KeyType>
void Learned_FIB<KeyType>::derive_boundaries(void) {
    double a, b;    // variable for slope and bias
    double p, err;  // variable for error calculation

    // B <- {(x_0, y_0)}
    boundaries.push_back(std::make_pair(splitted_dataset[0], starting_point));

    // l, r: left and right boundary of a line segment
    uint64_t l = 0, r = 2;
    while (r < splitted_dataset.size()) {
        if (splitted_dataset[r] == splitted_dataset[l]) {  // handling duplicate keys
            while (splitted_dataset[r] == splitted_dataset[l]) {
                r++;
            }
            r++;
            continue;
        }

        // Derive a line's (slope, bias) passing through (x_l, l) and (x_r, r)
        a = static_cast<double>(r - l) / static_cast<double>(splitted_dataset[r] - splitted_dataset[l]);
        b = l - a * static_cast<double>(splitted_dataset[l]);

        // Examine the error between x_(l+1) and x_(r-1)
        for (uint64_t i = l + 1; i < r; ++i) {
            p = a * static_cast<double>(splitted_dataset[i]) + b;  // compute the y-value on the line for the x-value of x_i
            double err = p > static_cast<double>(i) ? p - static_cast<double>(i) : static_cast<double>(i) - p;
            if (err >= error_threshold) {
                // The error is geq than the error_threshold
                // Append (x_(r-1), r-1) to B
                boundaries.push_back(std::make_pair(splitted_dataset[r - 1], r - 1 + starting_point));

                l = r - 1;
                break;
            }
        }
        r++;
    }

    // insert last point if not inserted
    if (splitted_dataset.back() != boundaries.back().first) {
        // std::cout << "insert last point\t";
        boundaries.push_back(std::make_pair(splitted_dataset.back(), splitted_dataset.size() - 1 + starting_point));
        return;
    }
    uint64_t cnt = 0;
    uint64_t idx = splitted_dataset.size() - 1;
    while (splitted_dataset[idx] == splitted_dataset.back()) {
        cnt++;
        idx--;
    }
    std::cout << "number of duplicates: " << cnt << "\t";
}

template <typename KeyType>
void Learned_FIB<KeyType>::train(const std::string& dataset_path, const std::string& model_path, double threshold) {
    // open dataset file
    std::ifstream dataset(dataset_path, std::ios::binary);
    if (!dataset.is_open()) {  // error check
        std::cerr << "[Learned_FIB.h] load dataset failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    // read dataset size
    dataset.read(reinterpret_cast<char*>(&dataset_size), sizeof(uint64_t));

    // record error boundary
    error_threshold = static_cast<uint64_t>(std::abs(threshold));

    // read data and start training
    KeyType prev_prefix = 0;
    starting_point = 0;
    KeyType key;
    for (uint64_t i = 0; i < dataset_size; ++i) {
        // push dataset
        dataset.read(reinterpret_cast<char*>(&key), sizeof(KeyType));

        // when prefix becomes different, derive pwl, train nn
        if ((calculate_prefix(key) != prev_prefix) || (i == dataset_size - 1)) {
            std::cout << prev_prefix << ": ";

            // derive boundaries (make PWL function)
            derive_boundaries();

            // train and save neural net
            NN<KeyType>* nn = new NN<KeyType>();
            nn->train(boundaries.begin(), boundaries.end());
            check_max_error(nn);
            nn->save(model_path + "_" + std::to_string(prev_prefix));
            delete nn;

            prev_prefix = calculate_prefix(key);

            // clear splitted_dataset and boundaries
            std::vector<KeyType>().swap(splitted_dataset);
            std::vector<std::pair<KeyType, uint64_t>>().swap(boundaries);
            starting_point = i;
        }
        splitted_dataset.push_back(key);
    }
    dataset.close();

    // write model max error
    std::ofstream os(model_path + "_max_error", std::ios::out);
    if (os.is_open()) {
        os << std::to_string(model_max_error);
        os.close();
    } else {
        std::cerr << "[NN.h save] Cannot open " + model_path + "_max_error" << std::endl;
        exit(EXIT_FAILURE);
    }

    return;
}

template <typename KeyType>
uint64_t Learned_FIB<KeyType>::find(KeyType key) {
    return static_cast<uint64_t>(NeuralNetworks[calculate_prefix(key)]->inference(key));
}

template <typename KeyType>
void Learned_FIB<KeyType>::load(const std::string& model_path) {
    for (size_t i = 0; i < (size_t)(1 << 8); ++i) {
        std::unique_ptr<NN<KeyType>> nn(new NN<KeyType>());
        NeuralNetworks.push_back(std::move(nn));
        NeuralNetworks[i]->load(model_path + "_" + std::to_string(i));
    }
}

#endif  // LEARNED_FIB_H