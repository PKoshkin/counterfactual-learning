#pragma once

#include "utils.h"


double get_metric(const Pool& test_pool, const std::vector<int>& predicted_positions);
double accuracy(const std::vector<double>& answers, const std::vector<int>& predictions, double eps = 0.01);
