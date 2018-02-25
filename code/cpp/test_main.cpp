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


int main(int argc, char* argv[]) {
    uint16_t max_labels = 14000;
    uint32_t tests_num = 10;
    uint32_t random_seed = 111;
    uint16_t initial_size = 1000;
    uint16_t batch_size = 500;

    Pool pool = get_test_classification_pool("../krkopt.data.txt");

    auto pool_pair = train_test_split(pool, 0.7, random_seed);
    Pool train_pool = pool_pair.first;
    Pool test_pool = pool_pair.second;

    std::string log_filename = "test_US_results.txt";
    std::vector<std::vector<int>> permutations = get_permutations(
        tests_num,
        train_pool.size(),
        random_seed
    );

    SimpleClassification model(50, 2);


    std::unique_ptr<ActiveLearningAlgo> active_learning_algo;
    std::unique_ptr<BasePoolBasedActiveLearningStrategy> strategy;
    set_algo(
        argv[1],
        active_learning_algo,
        strategy,
        model,
        initial_size,
        batch_size,
        max_labels,
        log_filename,
        &accuracy
    );

    test_active_learning_algo(
        active_learning_algo.get(),
        train_pool,
        test_pool,
        permutations,
        0,
        tests_num
    );

    return 0;
}
