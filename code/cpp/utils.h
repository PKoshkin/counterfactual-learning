#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iostream>

typedef std::vector<std::vector<double>> Matrix;

class Pool {
public:
    const std::vector<int> POSITIONS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100};
    Matrix factors;
    std::vector<int> positions;
    std::vector<double> metrics;
    std::vector<double> probas;
    Pool() = default;
    void reserve(int size);
    int size() const;
    void assign(const Pool& pool, int begin, int end = -1);
    std::vector<Pool> split_by_positions() const;
};

const int FACTORS_LEN = 1052;

Pool get_pool(std::string file_name, int max_line = 30000, int start_size = 0);
std::pair<Pool, Pool> train_test_split(const Pool& pool, double train_share);
