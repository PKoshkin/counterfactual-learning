#include <algorithm>
#include <iostream>

#include "metric.h"


double get_metric(const Pool& test_pool, const std::vector<int>& predicted_positions) {
    double result_metric = 0;

    for (int ind = 0; ind < test_pool.size(); ++ind)
        if (test_pool.positions[ind] == predicted_positions[ind]) {
            result_metric += test_pool.metrics[ind] / test_pool.probas[ind];
        }
    return result_metric / test_pool.size();
}


double accuracy(const std::vector<double>& answers, const std::vector<int>& predictions, double eps) {
    int right_cnt = 0;
    for (int i = 0; i < answers.size(); ++i)
        if (answers[i] - eps < predictions[i] && predictions[i] < answers[i] + eps)
            right_cnt++;
    return static_cast<double>(right_cnt) / answers.size();
}
