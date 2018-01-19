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

std::vector<int> get_positinos(const std::vector<double>& predicted_scores) {
    int size = predicted_scores.size() / POSITIONS.size();
    std::vector<int> result(size);

    for (int ind = 0; ind < size; ++ind) {
	auto begin_section = predicted_scores.begin() + ind * POSITIONS.size();
	auto end_section = predicted_scores.begin() + (ind + 1) * POSITIONS.size();

	int max_pos_ind = std::max_element(begin_section, end_section) - begin_section;
	result[ind] = POSITIONS[max_pos_ind];
    }

    return result;
}
