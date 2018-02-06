#pragma once


#include "counterfactural_model.h"
#include "active_learning_strategies.h"

#include <list>
#include <algorithm>


template<class Model>
class PoolBasedActiveLearningAlgo {
private:
    CounterfacturalModel<Model> model;
    std::list<Object> unlabeled_pool;
    Pool labeled_pool;
    uint16_t max_labels;
    uint16_t batch_size;
    BasePoolBasedActiveLearningStrategy<Model>* strategy;
public:
    // static std::string name = "Pool-based";
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
        this->max_labels = pool.size();

    std::vector<int> permutation = get_permutation(pool.size());
    for (int index = initial_size; index < permutation.size(); ++index)
        unlabeled_pool.push_back(pool.get(permutation[index]));

    labeled_pool.assign(pool, permutation.begin(), permutation.begin() + initial_size);
    labeled_pool.reserve(max_labels);
}


template<class Model>
CounterfacturalModel<Model> PoolBasedActiveLearningAlgo<Model>::train() {
    std::cout << "\nStart active learning train" << std::endl;
    model.fit(labeled_pool);
    while (labeled_pool.size() < max_labels) {
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

        model.fit(labeled_pool);
    }


    std::cout << "\nEnd active learning train" << std::endl;
    return model;
}
