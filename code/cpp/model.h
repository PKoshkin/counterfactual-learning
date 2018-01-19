#pragma once

#include "utils.h"

class Model {
public:
    std::vector<double> predict(const Matrix& features);
    void fit(const Matrix& features, const std::vector<double>& scores);
    double loss(const Matrix& features, std::vector<double> score);
    Model(double lr, double reg_lambda, int batch_size, int iterations_number);
private:
    double lr;
    double reg_lambda;
    int batch_size;
    int iterations_number;
    std::vector<double> weights;
    std::vector<double> get_gradient(const std::vector<double>& features, double score);
};