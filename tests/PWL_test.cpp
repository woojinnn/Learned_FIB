#include "../utils/PWL.h"

int main() {
    PWL<uint32_t> pwl;
    pwl.train("../utils/data/dataset", 0.5);
    pwl.save_boundaries("../utils/data/boundaries");
}