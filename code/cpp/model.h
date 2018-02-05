#pragma once

#include "utils.h"


class BaseModel {
public:
    virtual std::vector<double> predict(const Matrix& features) const = 0;
    virtual void fit(const Matrix& features, const std::vector<double>& scores) = 0;
private:
    virtual std::vector<double> get_gradient(const std::vector<double>& features, double score) = 0;
};


class LogisticRegression : public BaseModel{
public:
    virtual std::vector<double> predict(const Matrix& features) const;
    virtual void fit(const Matrix& features, const std::vector<double>& scores);
    double loss(const Matrix& features, std::vector<double> score);
    LogisticRegression(double lr, double reg_lambda, int batch_size, int iterations_number, double gradient_clip);
private:
    double lr;
    double reg_lambda;
    int batch_size;
    int iterations_number;
    double gradient_clip;
    std::vector<double> weights;

    virtual std::vector<double> get_gradient(const std::vector<double>& features, double score);
};


// TODO: add LinearRegression
