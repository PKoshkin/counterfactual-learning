#include "active_learning_algo.h"

#include <algorithm>


PoolBasedActiveLearningAlgo::PoolBasedActiveLearningAlgo(
        CounterfacturalModel* model,
        BasePoolBasedActiveLearningStrategy* strategy,
        uint16_t initial_size,
        uint16_t batch_size,
        uint16_t max_labels)
    : model(model), strategy(strategy), batch_size(batch_size), max_labels(max_labels), initial_size(initial_size) {}


std::string PoolBasedActiveLearningAlgo::name() {
    std::string result = "";

    result += "pool-based active learning algorithm using ";
    result += strategy->name();
    result += " strategy";
    if (max_labels == 0)
        result += " and whole pool";
    else
        result += " and " + std::to_string(max_labels) + " queries max";

    return result;
}


CounterfacturalModel* PoolBasedActiveLearningAlgo::train(const Pool& train_pool) {
    uint16_t current_max_labels = max_labels;
    if (max_labels == 0 || max_labels > train_pool.size())
        current_max_labels = train_pool.size();

    std::list<Object> unlabeled_pool;
    Pool labeled_pool;

    std::vector<int> permutation = get_permutation(train_pool.size());
    for (int index = initial_size; index < permutation.size(); ++index)
        unlabeled_pool.push_back(train_pool.get(permutation[index]));

    labeled_pool.assign(train_pool, permutation.begin(), permutation.begin() + initial_size);
    labeled_pool.reserve(current_max_labels);

    std::cout << "\nStart active learning train" << std::endl;
    if (!(strategy->is_model_free()))
        model->fit(labeled_pool);

    while (labeled_pool.size() < current_max_labels) {
        std::list<std::pair<std::list<Object>::iterator, double>> batch;
        uint16_t curr_batch_size = std::min(batch_size, uint16_t(max_labels - labeled_pool.size()));
        for (auto unlabeled_obj = unlabeled_pool.begin(); unlabeled_obj != unlabeled_pool.end(); ++unlabeled_obj) {
            double score = strategy->get_score(model, unlabeled_pool, labeled_pool, *unlabeled_obj);
            bool suit = false;

            for (auto it = batch.begin(); it != batch.end(); ++it)
                if (it->second < score) {
                    batch.insert(it, {unlabeled_obj, score});
                    suit = true;
                    break;
                }

            if (suit && batch.size() > curr_batch_size)
                batch.pop_back();
            if (!suit && batch.size() < curr_batch_size)
                batch.push_back({unlabeled_obj, score});
        }

        for (auto it: batch) {
            labeled_pool.push_back(*(it.first));
            unlabeled_pool.erase(it.first);
        }

        if (!strategy->is_model_free())
            model->fit(labeled_pool);
    }

    if (strategy->is_model_free())
        model->fit(labeled_pool);

    std::cout << "\nEnd active learning train" << std::endl;

    return model;
}
