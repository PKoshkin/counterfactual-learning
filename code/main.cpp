#include <iostream>

#include "metric.h"
#include "model.h"


int main() {
    char pool_path[100] = "../pool.json";
    Pool pool = get_pool(pool_path, 90000);
    std::vector<int> positions_counter(POSITIONS.size(), 0);

    for (int i = 0; i < pool.size(); ++i) {
	if (pool.positions[i] == 100)
	    positions_counter[10] += 1;
	else if (pool.positions[i] >= 0 and pool.positions[i] <= 9)
	    positions_counter[pool.positions[i]] += 1;
	else
	    std::cout << pool.positions[i] << std::endl;
    }

    std::cout << "Pool stats.\nPositions:" << std::endl;

    int sum = 0;
    for (int i = 0; i < POSITIONS.size(); ++i) {
	std::cout << POSITIONS[i] << ": " << positions_counter[i] << std::endl;
	sum += positions_counter[i];
    }
    std::cout << "Total amount: " << sum << std::endl;

    auto pool_pair = pool.train_test_split(0.75);
    Pool train_pool = pool_pair.first;
    Pool test_pool = pool_pair.second;

    std::cout << "Train size: " << train_pool.size() << std::endl;
    std::cout << "Test size:  " << test_pool.size() << std::endl;

    Model model(0.001, 1, 16, 50);
    model.fit(train_pool.factors, train_pool.metrics);
    std::vector<double> predicted_scores = model.predict(test_pool.factors);

    std::vector<int> predicted_positions = get_positinos(predicted_scores);
    for (int i = 0; i < 10; ++i)
	std::cout << "Predicted pos: " << predicted_positions[i] << " Real pos: " << test_pool.positions[i] << std::endl;

    std::cout << "Result metric: " << get_metric(test_pool, predicted_positions) << std::endl;
    return 0;
}
