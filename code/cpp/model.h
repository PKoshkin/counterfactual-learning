#pragma once

#include "utils.h"

class LinearRegression {
public:
    std::vector<double> predict(const Matrix& features);
    void fit(const Matrix& features, const std::vector<double>& scores);
    double loss(const Matrix& features, std::vector<double> score);
    LinearRegression(double lr, double reg_lambda, int batch_size, int iterations_number, double gradient_clip);
private:
    double lr;
    double reg_lambda;
    int batch_size;
    int iterations_number;
    double gradient_clip;
    std::vector<double> weights;

    std::vector<double> get_gradient(const std::vector<double>& features, double score);
};
