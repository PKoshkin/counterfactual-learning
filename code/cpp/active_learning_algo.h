#pragma once


#include "counterfactural_model.h"
#include "active_learning_strategies.h"

#include <list>
#include <algorithm>


template<class Model>
class PoolBasedActiveLearningAlgo {
private:
    CounterfacturalModel<Model> model;
    Pool unlabeled_pool;
    Pool labeled_pool;
    uint16_t max_labels;
    uint16_t batch_size;
    BasePoolBasedActiveLearningStrategy<Model>* strategy;
public:
    PoolBasedActiveLearningAlgo(const CounterfacturalModel<Model>& model,
                                const Pool& pool,
                                BasePoolBasedActiveLearningStrategy<Model>* strategy,
                                uint16_t initial_size = 100,
                                uint16_t batch_size = 5,
                                uint16_t max_labels = 0);
    CounterfacturalModel<Model> train();
};


template<class Model>
PoolBasedActiveLearningAlgo<Model>::PoolBasedActiveLearningAlgo(
        const CounterfacturalModel<Model>& model,
        const Pool& pool,
        BasePoolBasedActiveLearningStrategy<Model>* strategy,
        uint16_t initial_size,
        uint16_t batch_size,
        uint16_t max_labels) : model(model), strategy(strategy), batch_size(batch_size), max_labels(max_labels) {

    if (max_labels == 0)
        max_labels = pool.size();

    std::vector<int> permutation = get_permutation(pool.size());
    unlabeled_pool.assign(pool, permutation.begin() + initial_size, permutation.end());
    labeled_pool.assign(pool, permutation.begin(), permutation.begin() + initial_size);
    labeled_pool.reserve(max_labels);
}


template<class Model>
CounterfacturalModel<Model> PoolBasedActiveLearningAlgo<Model>::train() {
    while (labeled_pool.size() < max_labels) {
        std::list<std::pair<uint16_t, double>> batch;
        uint16_t curr_batch_size = std::min(batch_size, uint16_t(max_labels - labeled_pool.size()));

        for (uint16_t ind = 0; ind < unlabeled_pool.size(); ++ind) {
            double score = strategy->get_score(model, unlabeled_pool, labeled_pool, ind);
            bool suit = false;

            for (auto it = batch.begin(); it != batch.end(); ++it)
                if (it->second < score) {
                    batch.insert(it, {ind, score});
                    suit = true;
                    break;
                }

            if (suit && batch.size() > curr_batch_size)
                batch.pop_back();
            if (!suit && batch.size() < curr_batch_size)
                batch.push_back({ind, score});
        }

        for (auto it: batch) {
            Object obj = unlabeled_pool.get(it.first);
            unlabeled_pool.erase(it.first);
            labeled_pool.push_back(obj);
        }

        model.fit(labeled_pool);
    }

    return model;
}
