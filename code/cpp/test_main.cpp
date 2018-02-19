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
    uint16_t max_labels = 2000;
    uint32_t permutations_num = 10;
    uint32_t random_seed = 111;
    uint16_t initial_size = 1000;
    uint16_t batch_size = 100;

    Pool pool = get_test_classification_pool("../krkopt.data.txt");

    auto pool_pair = train_test_split(pool, 0.7, random_seed);
    Pool train_pool = pool_pair.first;
    Pool test_pool = pool_pair.second;
    std::vector<std::vector<int>> permutations = get_permutations(
        permutations_num,
        train_pool.size(),
        random_seed
    );

    SimpleClassification model(50, 18);
    // model.fit(train_pool);
    /*
    PoolBasedUncertaintySamplingStrategy strategy;
    PoolBasedActiveLearningAlgo active_learning_algo(
        &model,
        &strategy,
        initial_size,
        batch_size,
        max_labels,
        "test_US_results.txt",
        &accuracy
    );
    */
    PoolBasedPassiveLearningAlgo active_learning_algo(&model, max_labels);
    test_active_learning_algo(
        &active_learning_algo,
        train_pool,
        test_pool,
        permutations,
        0,
        permutations_num
    );
    return 0;
}
