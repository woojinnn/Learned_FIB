#ifndef PWL_H
#define PWL_H

#include <fstream>
#include <iostream>
#include <utility>  // pair
#include <vector>

template <typename KeyType>
class PWL {
   private:
    std::vector<KeyType> dataset;
    std::vector<std::pair<KeyType, uint64_t>> boundaries;
    void load_datset(const std::string& path);

   public:
    PWL(){};
    void train(const std::string& path, double threshold);
    void save_boundaries(const std::string& path);
};

template <typename KeyType>
void PWL<KeyType>::load_datset(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        std::cerr << "[PWL.cpp] load dataset failed" << std::endl;
        exit(0);
    }
    is.seekg(0, is.end);
    int length = (int)is.tellg() / sizeof(KeyType);
    is.seekg(0, is.beg);

    KeyType tmp;
    for (int i = 0; i < length; ++i) {
        is.read((char*)(&tmp), sizeof(KeyType));
        dataset.push_back(tmp);
    }

    is.close();
}

template <typename KeyType>
void PWL<KeyType>::train(const std::string& path, double threshold) {
    // load dataset
    load_datset(path);

    double a, b;    // variable for slope and bias
    double p, err;  // variable for error calculation

    // B <- {(x_0, 0)}
    boundaries.push_back(std::make_pair(dataset[0], 0));

    // l, r: left and right boundary of a line segment
    uint64_t l = 0, r = 2;
    while (r < dataset.size()) {
        // Derive a line passing through (x_l, l) and (x_r, r)
        a = static_cast<double>(r - l) / static_cast<double>(dataset[r] - dataset[l]);
        b = l - a * static_cast<double>(dataset[l]);

        // Examine the error between x_(l+1) and x_(r-1)
        for (uint64_t i = l + 1; i < r; ++i) {
            p = a * static_cast<double>(dataset[i]) + b;  // compute the y-value on the line for the x-value of x_i
            double err = p > i ? p - static_cast<double>(i) : static_cast<double>(i) - p;
            if (err > threshold) {
                // The error is larger than the threshold
                // Append (x_(r-1), r-1) to B
                boundaries.push_back(std::make_pair(dataset[r - 1], r - 1));

                l = r - 1;
                break;
            }
        }
        r++;
    }

    boundaries.push_back(std::make_pair(dataset[dataset.size() - 1], dataset.size() - 1));  // last point
}

template <typename KeyType>
void PWL<KeyType>::save_boundaries(const std::string& path) {
    std::ofstream fout;
    fout.open(path, std::ios::out | std::ios::binary);

    if (fout.is_open()) {
        for (int i = 0; i < boundaries.size(); ++i) {
            fout.write((const char*)(&(boundaries[i].first)), sizeof(KeyType));
            fout.write((const char*)(&(boundaries[i].second)), sizeof(uint64_t));
        }
        fout.close();
    } else {
        std::cerr << "[PWL.cpp] save boundareis failed" << std::endl;
        exit(0);
    }
}

#endif  // PWL_H