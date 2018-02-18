#include <iostream>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <limits>

#include "active_learning_algo.h"
#include "metric.h"
#include "model.h"
#include "test_model.h"
#include "al_utils.h"


int main() {
    uint16_t max_labels = 9000;
    Pool pool = get_test_classification_pool("../krkopt.data.txt");

    auto pool_pair = train_test_split(pool, 0.7, 111);
    Pool train_pool = pool_pair.first;
    Pool test_pool = pool_pair.second;
    std::vector<std::vector<int>> permutations(1);
    permutations[0] = get_permutation(train_pool.size());

    SimpleClassification model(50, 18);
    PoolBasedUncertaintySamplingStrategy strategy;
    PoolBasedActiveLearningAlgo active_learning_algo(&model, &strategy, 5000, 400, max_labels);
    // PoolBasedPassiveLearningAlgo active_learning_algo(&model, max_labels);
    test_active_learning_algo(
        &active_learning_algo,
        train_pool,
        test_pool,
        permutations,
        0, 1, "test_US_results.txt", 0
    );
    std::vector<int> predictions = model.predict(test_pool);
    std::cout << accuracy(test_pool.metrics, predictions) << std::endl;
    return 0;
}
