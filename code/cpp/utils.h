#pragma once

#include "pool.h"
#include <vector>
#include <string>
#include <memory>
#include <iostream>


const int FACTORS_LEN = 1052;

Pool get_pool(std::string file_name, int max_line = 30000, int start_size = 0);
std::pair<Pool, Pool> train_test_split(const Pool& pool, double train_share);

std::vector<int> get_permutation(int size);
std::vector<std::vector<int>> get_permutations(
        uint32_t permutations_number,
        uint32_t permutations_size,
        uint32_t random_seed = 0
);

void softmax(std::vector<double>& array);
