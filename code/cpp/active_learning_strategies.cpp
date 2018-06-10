#include "active_learning_strategies.h"

#include <cmath>
#include <algorithm>


double calculate_multiply(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0;
    for (uint32_t ind = 0; ind < a.size(); ++ind)
        result += a[ind] * b[ind];
    return result;
}

std::string PoolBasedUncertaintySamplingStrategy::name() {
    return "uncertainty sampling";
}


bool PoolBasedUncertaintySamplingStrategy::is_model_free() {
    return false;
}


void PoolBasedUncertaintySamplingStrategy::initialize(
    const Pool& train_pool,
    const std::vector<int>& permutation,
    int labeled_pool_size
) {}


void PoolBasedUncertaintySamplingStrategy::update(
    const Pool& train_pool,
    const std::vector<int>& batch,
    const std::list<int>& unlabeled_indexes
) {}


double PoolBasedUncertaintySamplingStrategy::get_score(
    CounterfacturalModel* current_model,
    const Pool& train_pool,
    int index
) {
    std::vector<double> probas = current_model->predict_proba(train_pool.get(index));

    return 1 - *std::max_element(probas.begin(), probas.end());
}


std::string PoolBasedDiversity::name() {
    return "diversity with " + std::to_string(seen_labeled_objects_share) + " share";
}


bool PoolBasedDiversity::is_model_free() {
    return true;
}


PoolBasedDiversity::PoolBasedDiversity(double seen_labeled_objects_share)
    : seen_labeled_objects_share(seen_labeled_objects_share) {}


double PoolBasedDiversity::cosin_similarity(
    const Matrix& factors,
    uint32_t ind_a,
    uint32_t ind_b
) {
    if (object_norms[ind_a] == -1)
        object_norms[ind_a] = std::sqrt(calculate_multiply(factors[ind_a], factors[ind_a]));
    if (object_norms[ind_b] == -1)
        object_norms[ind_b] = std::sqrt(calculate_multiply(factors[ind_b], factors[ind_b]));

    double denominator = object_norms[ind_a] * object_norms[ind_b];
    double numerator = calculate_multiply(factors[ind_a], factors[ind_b]);
    return numerator / denominator;
}


void PoolBasedDiversity::initialize(
        const Pool& train_pool,
        const std::vector<int>& permutation,
        int labeled_pool_size) {
    current_scores.resize(train_pool.size(), -2);
    object_norms.resize(train_pool.size(), -1);

    for (
        int unlabeled_permutation_ind = labeled_pool_size;
        unlabeled_permutation_ind < permutation.size();
        ++unlabeled_permutation_ind
    ) {

        int unlabeled_ind = permutation[unlabeled_permutation_ind];

        for (
            int labeled_ind = 0;
            labeled_ind < labeled_pool_size * seen_labeled_objects_share;
            ++labeled_ind
        ) {
            double score = cosin_similarity(
                train_pool.factors,
                unlabeled_ind,
                permutation[labeled_ind]
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
            double score = cosin_similarity(
                train_pool.factors,
                unlabeled_ind,
                batch[i]
            );
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

/*
PoolBasedQBC::PoolBasedQBC(std::vector<CounterfacturalModel*>&& committee)
    : committee(committee) {}


bool PoolBasedQBC::is_model_free() {return true;}


std::string PoolBasedQBC::name() {return "QBC";}


void PoolBasedQBC::initialize(
    const Pool& train_pool,
    const std::vector<int>& permutation,
    int labeled_pool_size
) {
    committee_probas = std::vector<std::vector<std::vector<double>>>(
        train_pool.size(),
        std::vector<std::vector<double>>(
            committee.size(),
            std::vector<double>(train_pool.POSITIONS.size())
        )
    );
    iteration_ind = 0;

    for (int model_ind = 0; model_ind < committee.size(); ++model_ind) {
        auto begin = permutation.begin() + model_ind * labeled_pool_size / committee.size();
        int end_ind = (model_ind + 1) * labeled_pool_size / committee.size();
        if (model_ind == committee.size() - 1)
            end_ind = permutation.size();
        auto end = permutation.begin() + end_ind;
        train_pools[model_ind].assign(train_pool, begin, end);
        if (model_ind == committee.size() - 1) {
            end = permutation.begin() + labeled_pool_size / committee.size();
            train_pools[model_ind].push_back(train_pool, permutation.begin(), end);
        }

        committee[model_ind]->fit(train_pools[model_ind]);
    }
}


void PoolBasedQBC::update(
    const Pool& train_pool,
    const std::vector<int>& batch,
    const std::list<int>& unlabeled_indexes
) {
    int model_ind = iteration_ind % committee.size();
    train_pools[model_ind].push_back(train_pool, batch.begin(), batch.end());
    for (auto obj_ind: unlabeled_indexes) {

        train_pool
    }
}
*/
