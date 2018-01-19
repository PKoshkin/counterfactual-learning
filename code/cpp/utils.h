#pragma once

#include <vector>
#include <string>
#include <memory>

typedef std::vector<std::vector<double>> Matrix;

class Pool {
public:
    short is_train; // 1 - train, 0 - test, -1 - not splited
    std::vector<std::vector<double>> factors;
    std::vector<int> positions;
    std::vector<double> metrics;
    std::vector<double> probas;
    Pool() : is_train(-1) {};
    void reserve(int size);
    int size() const;
    std::pair<Pool, Pool> train_test_split(double train_share);
    Matrix get_train_features();
    Matrix get_test_features();
};

const int FACTORS_LEN = 1052;
const std::vector<int> POSITIONS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100};

Pool get_pool(std::string file_name, int max_line = 30000, int start_size = 0);
