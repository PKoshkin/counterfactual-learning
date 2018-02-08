#pragma once

#include "utils.h"


class BaseModel {
public:
    virtual double predict(const std::vector<double>& features) const = 0;
    virtual void fit(const Matrix& features, const std::vector<double>& scores) = 0;
};


class LinearRegression : public BaseModel {
public:
    virtual double predict(const std::vector<double>& features) const;
    virtual void fit(const Matrix& features, const std::vector<double>& scores);
    virtual double loss(const Matrix& features, std::vector<double> score) = 0;
    LinearRegression(double lr, double reg_lambda, int batch_size, int iterations_number, double gradient_clip);
protected:
    double lr;
    double reg_lambda;
    int batch_size;
    int iterations_number;
    double gradient_clip;
    std::vector<double> weights;

    virtual std::vector<double> get_gradient(const std::vector<double>& features, double score) = 0;
};


class LogisticRegression : public LinearRegression {
public:
    using LinearRegression::LinearRegression;
    virtual double loss(const Matrix& features, std::vector<double> score);
private:
    virtual std::vector<double> get_gradient(const std::vector<double>& features, double score);
};


class RidgeRegression : public LinearRegression {
public:
    using LinearRegression::LinearRegression;
    virtual double loss(const Matrix& features, std::vector<double> score);
private:
    virtual std::vector<double> get_gradient(const std::vector<double>& features, double score);
};
