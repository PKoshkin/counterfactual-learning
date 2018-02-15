#pragma once

#include "active_learning_algo.h"
#include "pool.h"


void test_active_learning_algo(
        ActiveLearningAlgo* active_learning_algo,
        const Pool& train_pool,
        const Pool& test_pool,
        const std::vector<std::vector<int>>& permutations,
        uint16_t start_permutaion_ind,
        uint16_t permutations_num,
        std::string file_name,
        uint16_t pos_to_print_number = 5);
