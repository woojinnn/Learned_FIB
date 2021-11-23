#include "../include/Learned_FIB.h"

#include <string.h>

#include <fstream>
#include <iostream>

template <typename KeyType>
uint64_t validate_model(const std::string& data_path, Learned_FIB<KeyType>& lf) {
    std::ifstream is(data_path, std::ios::binary);
    if (!is.is_open()) {
        std::cout << "[Learned_FIB.cpp - validate_model]: Cannot open file!";
        exit(-1);
    }

    uint64_t data_size;
    is.read(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));

    KeyType data_key;
    uint64_t max_err = 0;
    for (uint64_t i = 0; i < data_size; ++i) {
        is.read(reinterpret_cast<char*>(&data_key), sizeof(KeyType));

        uint64_t pred = lf.find(data_key);
        uint64_t err = pred > i ? pred - i : i - pred;
        if (err > max_err) {
            max_err = err;
        }
    }
    std::cout << "max err: " << max_err << std::endl;
    return max_err;
}

void print_argument_usage(void) {
    std::cerr << "== Required Arguments ==" << std::endl;
    std::cerr << "argv[0]: execute file" << std::endl;
    std::cerr << "argv[1]: data_path" << std::endl;
    std::cerr << "argv[2]: model_path" << std::endl;
    std::cerr << "argv[3]: Key Type (32 or 64)" << std::endl;
    std::cerr << "argv[4]: prefix" << std::endl;
    std::cerr << "argv[5]: error bound" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "We need 5 arguments, but you gave " << std::to_string(argc) << " arguments" << std::endl;
        print_argument_usage();
        exit(-1);
    }

    // ./Learned_FIB /home/woojin/SOSD/data/dna_uint32 /home/woojin/Learned_FIB/tests/test_model/branching/8/nn_trained 32 8 32
    std::string data_path = "/home/woojin/SOSD/data/dna_uint32";
    std::string model_path = "/home/woojin/Learned_FIB/tests/test_model/branching/8/nn_trained";
    // std::string data_path = argv[1];
    // std::string model_path = argv[2];

    unsigned int keytype = std::stoul(argv[3]);
    unsigned int prefix = std::stoul(argv[4]);
    double error_bound = std::stod(argv[5]);

    std::cout << "Start training with:" << std::endl;
    std::cout << "\tData path:\t" << argv[1] << std::endl;
    std::cout << "\tModel path:\t" << argv[2] << std::endl;
    std::cout << "\tKeyType:\tuint" << argv[3] << std::endl;
    std::cout << "\tPrefix:\t" << argv[4] << std::endl;
    std::cout << "\tError bound:\t" << argv[5] << std::endl;

    if (keytype == 32) {
        Learned_FIB<uint32_t> lf(prefix);
        lf.train(data_path, model_path, error_bound);
    } else if (keytype == 64) {
        Learned_FIB<uint64_t> lf(prefix);
        lf.train(data_path, model_path, error_bound);
    } else {
        std::cerr << "KeyType should be 32 or 64" << std::endl;
        std::cerr << "Your input was " << argv[3] << std::endl;
        exit(-1);
    }

    std::cout << "Done" << std::endl;

    Learned_FIB<uint32_t> lf(prefix);
    lf.load(model_path);
    validate_model<uint32_t>(data_path, lf);
}