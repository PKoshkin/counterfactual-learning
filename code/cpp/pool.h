#pragma once

#include <vector>


typedef struct Object {
    int position;
    double metric;
    double proba;
    std::vector<double> factors;
    Object(int position, double metric, double proba, const std::vector<double> factors);
} Object;


typedef std::vector<std::vector<double>> Matrix;


class Pool {
public:
    Matrix factors;
    std::vector<int> positions;
    std::vector<double> metrics;
    std::vector<double> probas;

    const std::vector<int> POSITIONS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100};
    Pool() = default;
    void reserve(int size);
    void resize(int size);
    int size() const;
    void assign(const Pool& pool, int begin, int end = -1);
    void assign(const Pool& pool, std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end);
    std::vector<Pool> split_by_positions() const;
    void push_back(const Object& obj);
    void erase(int index);
    void set(int index, const Object& obj);
    void set(int index, int position, double metric, double proba, const std::vector<double>& factors);
    Object get(int index) const;
};
