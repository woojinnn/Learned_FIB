#include "../include/Learned_FIB.h"

#include <string.h>

#include <fstream>
#include <iostream>

template <typename KeyType>
uint64_t validate_model(const std::string& data_path, Learned_FIB<KeyType>& lf) {
    std::ifstream is(data_path, std::ios::binary);

    uint64_t data_size;
    is.read(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));
    std::cout << data_size << std::endl;

    KeyType data_key;
    KeyType prev_key = 9999;
    uint64_t max_err = 0;
    for (uint64_t i = 0; i < data_size; ++i) {
        is.read(reinterpret_cast<char*>(&data_key), sizeof(KeyType));

        if (data_key != prev_key) {
            uint64_t pred = lf.find(data_key);
            uint64_t err = pred > i ? pred - i : i - pred;
            if (err > max_err) {
                max_err = err;
            }
        }

        prev_key = data_key;
    }
    std::cout << "max err: " << max_err << std::endl;
    return max_err;
}

int main(void) {
    std::string data_path = "../data/dna_uint64";

    Learned_FIB<uint64_t> lf;
    lf.train(data_path, 16);
    // lf.load("./test_model/branching/8/nn_trained");

    validate_model<uint64_t>(data_path, lf);
    lf.save("./test_model/branching/16/nn_trained");
}