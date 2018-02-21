#pragma once

#include "pool.h"
#include <vector>
#include <string>
#include <memory>
#include <iostream>


const int FACTORS_LEN = 1052;
const std::vector<std::string> KRKOPT_ANSWERS = {
    "draw", "zero", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifthteen", "sixteen"
};


Pool get_test_classification_pool(std::string file_name, int start_size = 28056);
Pool get_pool(std::string file_name, int max_line = 30000, int start_size = 0);
std::pair<Pool, Pool> train_test_split(
    const Pool& pool,
    double train_share,
    uint32_t random_seed = -1
);

std::vector<int> get_permutation(int size);
std::vector<std::vector<int>> get_permutations(
        uint32_t permutations_number,
        uint32_t permutations_size,
        uint32_t random_seed = 0
);

void softmax(std::vector<double>& array);

uint32_t get_uint_param(
    int argc,
    char* argv[],
    uint16_t index,
    uint32_t default_value,
    const char name[]
);
