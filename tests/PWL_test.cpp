#include "../include/PWL.h"

int main() {
    PWL<uint32_t> pwl;
    pwl.train("../include/data/dataset", 0.5);
    pwl.save_boundaries("../include/data/boundaries");
}