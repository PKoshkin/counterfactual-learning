#include <algorithm>
#include <cmath>
#include <iostream>

#include "model.h"


double LogisticRegression::predict(const std::vector<double>& features) const {
    double result = 0;

    for (int i = 0; i < features.size(); ++i)
        result += weights[i] * features[i];

    return result;
}


void LogisticRegression::fit(const Matrix& features, const std::vector<double>& scores) {
    weights.resize(features[0].size(), 0);

    for (int iter = 0; iter < iterations_number; ++iter) {
        std::vector<int> random_permutation = get_permutation(features.size());

        for (int batch_start = 0; batch_start < features.size(); batch_start += batch_size) {
            std::vector<double> gradient(weights.size(), 0);
            for (int index = batch_start; index < std::min(batch_start + batch_size, int(features.size())); ++index) {
                int pool_ind = random_permutation[index];
                std::vector<double> curr_grad = get_gradient(features[pool_ind], scores[pool_ind]);
                for (int feature_index = 0; feature_index < gradient.size(); ++feature_index)
                    gradient[feature_index] += curr_grad[feature_index];
            }

            for (int feature_index = 0; feature_index < gradient.size(); ++feature_index)
                weights[feature_index] -= lr * gradient[feature_index];

        }
        // std::cout << "weights: " << weights[255] << " " << weights[1051] << std::endl;
        // std::cout << "loss: " << loss(features, scores) << std::endl;
    }
}


LogisticRegression::LogisticRegression(double lr, double reg_lambda, int batch_size, int iterations_number, double gradient_clip)
    : lr(lr), reg_lambda(reg_lambda), batch_size(batch_size), iterations_number(iterations_number), gradient_clip(gradient_clip) {}


std::vector<double> LogisticRegression::get_gradient(const std::vector<double>& features, double score) {
    std::vector<double> result = features;

    double margin_exp = std::exp(score * predict(features));
    double coef = -score / (1 + margin_exp);

    for (int i = 0; i < result.size(); ++i) {
	result[i] *= coef;
	result[i] += 2 * reg_lambda * weights[i];
    }

    // std::cout << coef << " " << score << " " << -score * predict(features) << std::endl;
    return result;

    /*
    double prediction = predict(features);
    std::vector<double> result = features;
    for (int i = 0; i < features.size(); ++i) {
        result[i] *= 2 * (prediction - score);
        result[i] += 2 * reg_lambda * weights[i];

        if (result[i] > gradient_clip)
            result[i] = gradient_clip;
        if (result[i] < -gradient_clip)
            result[i] = -gradient_clip;

    }
    return result;*/
}


double LogisticRegression::loss(const Matrix& features, std::vector<double> scores) {
    double loss = 0;

    for (int i = 0; i < features.size(); ++i)
	loss += std::log(1 + std::exp(-scores[i] * predict(features[i])));

    for (int i = 0; i < weights.size(); ++i)
	loss += weights[i] * weights[i];

    return loss;
    /*
    double loss = 0;

    for (int i = 0; i < features.size(); ++i) {
        double prediction = predict(features[i]);
        loss += (scores[i] - prediction) * (scores[i] - prediction);
    }
    loss /= features.size();

    for (int i = 0; i < weights.size(); ++i)
         loss += weights[i] * weights[i];

    return loss;*/
}
