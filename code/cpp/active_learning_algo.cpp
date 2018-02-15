#include "active_learning_algo.h"

#include <algorithm>


PoolBasedActiveLearningAlgo::PoolBasedActiveLearningAlgo(
        CounterfacturalModel* model,
        BasePoolBasedActiveLearningStrategy* strategy,
        uint16_t initial_size,
        uint16_t batch_size,
        uint16_t max_labels)
        : strategy(strategy), batch_size(batch_size), max_labels(max_labels) {
    this->model = model;
    this->initial_size = initial_size;
}


std::string PoolBasedActiveLearningAlgo::name() {
    std::string result = "";

    result += "pool-based active learning algorithm using ";
    result += strategy->name();
    result += " strategy";
    if (max_labels == 0)
        result += " and whole pool";
    else
        result += " and " + std::to_string(max_labels) + " queries max";
    result += " with batch size " + std::to_string(batch_size);

    return result;
}


CounterfacturalModel* PoolBasedActiveLearningAlgo::train(const Pool& train_pool, const std::vector<int>& permutation) {
    uint16_t current_max_labels = max_labels;
    if (max_labels == 0 || max_labels > train_pool.size())
        current_max_labels = train_pool.size();

    std::list<int> unlabeled_indexes;
    for (int index = initial_size; index < permutation.size(); ++index)
        unlabeled_indexes.push_back(permutation[index]);

    Pool labeled_pool;
    labeled_pool.assign(train_pool, permutation.begin(), permutation.begin() + initial_size);
    labeled_pool.reserve(current_max_labels);

    if (!(strategy->is_model_free()))
        model->fit(labeled_pool);

    strategy->initialize(train_pool, permutation, initial_size);
    std::cout << "\nStart active learning train" << std::endl;

    while (labeled_pool.size() < current_max_labels) {
        std::list<std::pair<std::list<int>::iterator, double>> batch;
        uint16_t curr_batch_size = std::min(batch_size, uint16_t(current_max_labels - labeled_pool.size()));
        for (auto unlabeled_ind = unlabeled_indexes.begin(); unlabeled_ind != unlabeled_indexes.end(); ++unlabeled_ind) {
            double score = strategy->get_score(model, train_pool, *unlabeled_ind);
            bool suit = false;

            for (auto it = batch.begin(); it != batch.end(); ++it)
                if (it->second < score) {
                    batch.insert(it, {unlabeled_ind, score});
                    suit = true;
                    break;
                }

            if (suit && batch.size() > curr_batch_size)
                batch.pop_back();
            if (!suit && batch.size() < curr_batch_size)
                batch.push_back({unlabeled_ind, score});
        }

        std::vector<int> batch_ind;
        batch_ind.reserve(batch.size());
        for (auto it: batch) {
            batch_ind.push_back(*(it.first));
            labeled_pool.push_back(train_pool.get(*(it.first)));
            unlabeled_indexes.erase(it.first);
        }
        strategy->update(train_pool, batch_ind, unlabeled_indexes);

        std::cout << "\nBatch added" << std::endl;

        if (!strategy->is_model_free())
            model->fit(labeled_pool);
    }

    if (strategy->is_model_free())
        model->fit(labeled_pool);

    std::cout << "\nEnd active learning train" << std::endl;

    return model;
}


std::string PoolBasedPassiveLearningAlgo::name() {
     std::string result = "";

    result += "pool-based passive learning algorithm choosing ";
    result += std::to_string(max_labels) + " queries";

    return result;
}


PoolBasedPassiveLearningAlgo::PoolBasedPassiveLearningAlgo(
        CounterfacturalModel* model,
        uint16_t max_labels) : max_labels(max_labels) {
    this->model = model;
}


CounterfacturalModel* PoolBasedPassiveLearningAlgo::train(const Pool& train_pool, const std::vector<int>& permutation) {
    Pool actual_train_pool;
    actual_train_pool.assign(train_pool, permutation.begin(), permutation.begin() + max_labels);
    model->fit(actual_train_pool);
    return model;
}
