#include "learning.h"
#include <iostream>

int main() {
    char pool_path[100] = "little_pool.json";
    Pool pool = get_pool(pool_path);
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

    BoosterHandle booster = train(train_pool);
    float result = test(test_pool, booster);
    XGBoosterFree(booster);
    std::cout << result << std::endl;
    return 0;
}
