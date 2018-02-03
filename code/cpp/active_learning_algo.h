#include "coutetfactural_model.h"
#include "active_learning_strategies.h"
#include <set>


template<class Model, class Strategy>
class PoolBasedActiveLearningAlgo {
private:
    CounterfacturalModel<Model> model;
    std::set<uint16_t> unlabeled_indexes;
    Pool labeled_pool;
    uint16_t max_labels;
    Pool pool;
    Strategy strategy;
public:
    PoolBasedActiveLearningAlgo(const Model& model,
                                const Pool& pool,
                                const Strategy& strategy);
    CounterfacturalModel<Model> train();
};


template<class Model, class Strategy>
PoolBasedActiveLearningAlgo<Model, Strategy>::PoolBasedActiveLearningAlgo(
        const Model& model,
        const Pool& pool,
        const Strategy& strategy,
        uint16_t initial_size,
        uint16_t max_labels = 0) : model(model), pool(pool), strategy(strategy), max_labels(max_labels) {
    if (max_labels == 0)
        max_labels = pool.size();
    std::vector<int> permutation = get_permutation(pool.size());
    unlabeled_indexes.insert(permutation.begin() + initial_size, permutation.end());
    labeled_pool.assign(pool, permutation, initial_size);
    labeled_pool.reserve(max_labels);
}


template<class Model, class Strategy>
CounterfacturalModel<Model> PoolBasedActiveLearningAlgo<Model, Strategy>::train() {
    while (labeled_indexes.size() < max_labels) {
        vector<uint16_t> batch = strategy.get_batch(pool, model, unlabeled_indexes, labeled_pool);
        for (auto ind: batch) {
            unlabeled_indexes.erase(ind);
            labeled_pool.push_back(pool.get(ind));
        }
        model.fit(labeled_pool);
    }

    return model;
}
