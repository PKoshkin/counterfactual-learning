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


void PoolBasedUncertaintySamplingStrategy::initialize(
        const Pool& train_pool,
        const std::vector<int>& indexes_permutation,
        int labeled_pool_size) {
}


void PoolBasedUncertaintySamplingStrategy::update(
        const Pool& train_pool,
        const std::vector<int>& batch,
        const std::list<int>& unlabeled_indexes) {
}


double PoolBasedUncertaintySamplingStrategy::get_score(
        CounterfacturalModel* current_model,
        const Pool& train_pool,
        int index) {
    std::vector<double> probas = current_model->predict_proba(train_pool.get(index));

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


PoolBasedDiversity::PoolBasedDiversity(double seen_labeled_objects_share)
    : seen_labeled_objects_share(seen_labeled_objects_share) {}


void PoolBasedDiversity::initialize(
        const Pool& train_pool,
        const std::vector<int>& indexes_permutation,
        int labeled_pool_size) {
    current_scores.resize(train_pool.size(), -2);

    for (int unlabeled_permutation_ind = labeled_pool_size;
            unlabeled_permutation_ind < indexes_permutation.size();
            ++unlabeled_permutation_ind) {

        int unlabeled_ind = indexes_permutation[unlabeled_permutation_ind];

        for (int labeled_ind = 0; labeled_ind < labeled_pool_size * seen_labeled_objects_share; ++labeled_ind) {
            double score = cosin_similarity(
                train_pool.factors[unlabeled_ind],
                train_pool.factors[indexes_permutation[labeled_ind]]
            );
            if (score > current_scores[unlabeled_ind])
                current_scores[unlabeled_ind] = score;
        }
    }
}


void PoolBasedDiversity::update(
        const Pool& train_pool,
        const std::vector<int>& batch,
        const std::list<int>& unlabeled_indexes) {
    for (int i = 0; i < batch.size() * seen_labeled_objects_share; ++i)
        for (auto unlabeled_ind: unlabeled_indexes) {
            double score = cosin_similarity(train_pool.factors[unlabeled_ind], train_pool.factors[batch[i]]);
            if (score > current_scores[unlabeled_ind])
                current_scores[unlabeled_ind] = score;
        }
}


double PoolBasedDiversity::get_score(
        CounterfacturalModel* current_model,
        const Pool& train_pool,
        int index) {
    return -current_scores[index];
}
