#include "../include/Learned_FIB.h"

int main() {
    Learned_FIB<uint32_t> lf;
    lf.train("../data/dna_uint32", 500);
    lf.save("./test_model/nn_trained");

    Learned_FIB<uint32_t> lf_new;
    lf_new.load("./test_model/nn_trained");

    std::cout << lf_new.get_error_threshold() << std::endl;

    std::cout << lf_new.find(354985874) << std::endl;  // 3520122
    std::cout << lf_new.find(2968539296) << std::endl; // 18819277
    std::cout << lf_new.find(4293754898) << std::endl; // 28542655
}