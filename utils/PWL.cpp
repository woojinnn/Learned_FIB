#include <fstream>
#include <iostream>
#include <utility>  // pair
#include <vector>

std::vector<uint32_t> points;
void load_datset() {
    std::ifstream is("data/dataset", std::ios::binary);
    if (!is) {
        std::cerr << "[PWL.cpp] load dataset failed" << std::endl;
        exit(0);
    }
    is.seekg(0, is.end);
    int length = (int)is.tellg() / sizeof(uint32_t);
    is.seekg(0, is.beg);

    uint32_t tmp;
    for (int i = 0; i < length; ++i) {
        is.read((char*)(&tmp), sizeof(uint32_t));
        points.push_back(tmp);
    }

    is.close();
}

class PWL {
   private:
    std::vector<uint32_t> boundaries;
    double threshold;

   public:
    PWL(double threshold) : threshold(threshold){};
    void train();
    void save_boundaries();
};

void PWL::train() {
    double a, b;    // variable for slope and bias
    double p, err;  // variable for error calculation

    // B <- {(x_0, 0)}
    boundaries.push_back(0);

    // l, r: left and right boundary of a line segment
    uint32_t l = 0, r = 2;
    while (r < points.size()) {
        // Derive a line passing through (x_l, l) and (x_r, r)
        a = static_cast<double>(r - l) / static_cast<double>(points[r] - points[l]);
        b = l - a * static_cast<double>(points[l]);

        // Examine the error between x_(l+1) and x_(r-1)
        for (uint32_t i = l + 1; i < r; ++i) {
            p = a * static_cast<double>(points[i]) + b;  // compute the y-value on the line for the x-value of x_i
            double err = p > i ? p - static_cast<double>(i) : static_cast<double>(i) - p;
            if (err > threshold) {
                // The error is larger than the threshold
                // Append (x_(r-1), r-1) to B
                boundaries.push_back(r - 1);

                l = r - 1;
                break;
            }
        }
        r++;
    }

    boundaries.push_back(points.size() - 1);  // last point
}

void PWL::save_boundaries() {
    std::ofstream fout;
    fout.open("data/boundaries", std::ios::out | std::ios::binary);

    if (fout.is_open()) {
        for (int i = 0; i < boundaries.size(); ++i) {
            fout.write((const char*)(&boundaries[i]), sizeof(uint32_t));
        }
        fout.close();
    } else {
        std::cerr << "[PWL.cpp] save boundareis failed" << std::endl;
        exit(0);
    }
}

int main() {
    load_datset();
    PWL pwl(0.5);
    pwl.train();
    pwl.save_boundaries();
}