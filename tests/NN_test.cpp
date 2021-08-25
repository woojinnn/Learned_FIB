#include "../include/NN.h"

int main() {
    std::vector<std::pair<uint32_t, uint64_t>> points;
    points.push_back(std::make_pair(0, 1));
    points.push_back(std::make_pair(2, 3));
    points.push_back(std::make_pair(5, 8));
    points.push_back(std::make_pair(10, 15));
    points.push_back(std::make_pair(22, 25));

    NN<uint32_t> test;
    test.train(points.begin(), points.end());
    std::cout << test.inference(0) << std::endl;
    std::cout << test.inference(2) << std::endl;
    std::cout << test.inference(4) << std::endl;
    std::cout << test.inference(5) << std::endl;
    std::cout << test.inference(10) << std::endl;
    std::cout << test.inference(22) << std::endl;
    std::cout << test.inference(9) << std::endl;
}