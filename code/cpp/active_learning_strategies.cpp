#include "active_learning_strategies.h"

#include <cmath>


double cosin_similarity(const std::vector<double>& a, const std::vector<double>& b) {
    double a_squares_sum = 0, b_squares_sum = 0, multiply = 0;
    for (int i = 0; i < a.size(); ++i) {
        a_squares_sum += a[i] * a[i];
        b_squares_sum += b[i] * b[i];
        multiply += a[i] * b[i];
    }

    return multiply / std::sqrt(a_squares_sum * b_squares_sum);
}


std::string PoolBasedUncertaintySamplingStrategy::name() {
    return "uncertainty sampling";
}


bool PoolBasedUncertaintySamplingStrategy::is_model_free() {
    return false;
}


double PoolBasedUncertaintySamplingStrategy::get_score(
        CounterfacturalModel* current_model,
        const std::list<Object>& unlabeled_pool,
        const Pool& labeled_pool,
        const Object& obj) {
    std::vector<double> probas = current_model->predict_proba(obj);

    double gini_score = 1;
    for (auto proba: probas)
        gini_score -= proba * proba;

    return gini_score;
}


std::string PoolBasedDiversity::name() {
    return "diversity";
}


bool PoolBasedDiversity::is_model_free() {
    return true;
}


double PoolBasedDiversity::get_score(
        CounterfacturalModel* current_model,
        const std::list<Object>& unlabeled_pool,
        const Pool& labeled_pool,
        const Object& obj) {
    double max = -2;
    for (int i = 0; i < labeled_pool.size(); ++i) {
        double similarity = cosin_similarity(obj.factors, labeled_pool.factors[i]);
        if (similarity > max)
            max = similarity;
    }
    return max;
}
