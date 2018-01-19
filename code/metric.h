#pragma once

#include "utils.h"


double get_metric(const Pool& test_pool, const std::vector<int>& predicted_positions);
std::vector<int> get_positinos(const std::vector<double>& predicted_scores);
