#pragma once

#include "utils.h"

typedef double Metric(const Pool& test_pool, const std::vector<int>& predictions);

double get_metric(const Pool& test_pool, const std::vector<int>& predictions);
double accuracy(const Pool& test_pool, const std::vector<int>& predictions);
