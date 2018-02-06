#pragma once


#include "counterfactural_model.h"
#include "active_learning_strategies.h"

#include <list>


class PoolBasedActiveLearningAlgo {
private:
    CounterfacturalModel model;
    std::list<Object> unlabeled_pool;
    Pool labeled_pool;
    uint16_t max_labels;
    uint16_t batch_size;
    BasePoolBasedActiveLearningStrategy* strategy;
public:
    // static std::string name = "Pool-based";
    PoolBasedActiveLearningAlgo(const CounterfacturalModel& model,
                                const Pool& pool,
                                BasePoolBasedActiveLearningStrategy* strategy,
                                uint16_t initial_size = 100,
                                uint16_t batch_size = 5,
                                uint16_t max_labels = 0);
    CounterfacturalModel train();
};
