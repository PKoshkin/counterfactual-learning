#pragma once

#include "utils.h"
#include <xgboost/c_api.h>

BoosterHandle train(const Pool& train_pool, int iteration_number = 200, int batch_size = 10);
float test(const Pool& test_pool, const BoosterHandle& booster, int batch_size = 10);
