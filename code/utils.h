#pragma once

#include <vector>
#include <string>
#include <memory>

typedef struct Object {
    std::vector<float> factors;
    int position;
    float metric;
    float proba;
    Object() = default;

    Object(const Object& obj)
	: factors(obj.factors), position(obj.position), metric(obj.metric), proba(obj.proba) {}

    Object(std::vector<float> factors, int position, float metric, float proba)
	: factors(factors), position(position), metric(metric), proba(proba) {}
} Object;

typedef std::vector<std::vector<float>> Matrix;

class Pool {
public:
    short is_train; // 1 - train, 0 - test, -1 - not splited
    std::vector<std::vector<float>> factors;
    std::vector<int> positions;
    std::vector<float> metrics;
    std::vector<float> probas;
    Pool() : is_train(-1) {};
    void reserve(int size);
    int size() const;
    std::pair<Pool, Pool> train_test_split(float train_share);
    Matrix get_train_features();
    Matrix get_test_features();
};

const int FACTORS_LEN = 1052;
const std::vector<int> POSITIONS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100};

Pool get_pool(std::string file_name, int max_line = 30000, int start_size = 0);
