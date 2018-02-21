#pragma once

#include "active_learning_algo.h"
#include "pool.h"


void test_active_learning_algo(
    ActiveLearningAlgo* active_learning_algo,
    const Pool& train_pool,
    const Pool& test_pool,
    const std::vector<std::vector<int>>& permutations,
    uint16_t start_permutaion_ind,
    uint16_t permutations_num
);
void set_algo(
    const char strategy_name[],
    std::unique_ptr<ActiveLearningAlgo>& active_learning_algo,
    std::unique_ptr<BasePoolBasedActiveLearningStrategy>& strategy,
    CounterfacturalModel& model,
    uint32_t initial_size,
    uint32_t batch_size,
    uint32_t max_labels,
    std::string log_filename,
    Metric* metric
);
