#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iostream>

typedef std::vector<std::vector<double>> Matrix;

typedef struct Object {
    int position;
    double metric;
    double proba;
    std::vector<double> factors;
    Object(int position, double metric, double proba, const std::vector<double> factors);
} Object;

class Pool {
public:
    const std::vector<int> POSITIONS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100};
    Matrix factors;
    std::vector<int> positions;
    std::vector<double> metrics;
    std::vector<double> probas;
    Pool() = default;
    void reserve(int size);
    void resize(int size);
    int size() const;
    void assign(const Pool& pool, int begin, int end = -1);
    void assign(const Pool& pool, std::vector<int> indexes, int end = -1);
    std::vector<Pool> split_by_positions() const;
    void push_back(const Object& obj);
    void set(int index, const Object& obj);
    void set(int index, int position, double metric, double proba, const std::vector<double>& factors);
    Object get(uint16_t index);
};

const int FACTORS_LEN = 1052;

Pool get_pool(std::string file_name, int max_line = 30000, int start_size = 0);
std::pair<Pool, Pool> train_test_split(const Pool& pool, double train_share);
std::vector<int> get_permutation(int size);
