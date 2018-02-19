#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <memory>

#include "metric.h"
#include "xgboost_model.h"
#include "counterfactural_model.h"
#include "active_learning_algo.h"
#include "al_utils.h"


int main(int argc, char* argv[]) {
    uint16_t test_results_sample_size = 30;

    uint16_t start_permutaion_ind;
    if (argc > 2)
        start_permutaion_ind = atoi(argv[2]);
    else
        start_permutaion_ind = 0;

    uint16_t permutations_num;
    if (argc > 3)
        permutations_num = atoi(argv[3]);
    else
        permutations_num = test_results_sample_size;

    uint16_t batch_size;
    if (argc > 4)
        batch_size = atoi(argv[4]);
    else
        batch_size = 400;

    uint16_t end_permutaiont_ind = start_permutaion_ind + permutations_num;
    uint16_t initial_size = 5000;
    uint16_t max_labels = 9000;
    std::string log_filename = "al_test_results.txt";

    char pool_path[100] = "../../pool.json";
    Pool pool = get_pool(pool_path, 90000);
    auto pool_pair = train_test_split(pool, 0.75);
    Pool train_pool = pool_pair.first;
    Pool test_pool = pool_pair.second;

    std::vector<std::vector<int>> permutations = get_permutations(
        test_results_sample_size,
        train_pool.size(),
        0
    );

    XGBoostModel base_model(50);
    PositionToFeaturesModel model(&base_model, train_pool.POSITIONS);

    std::unique_ptr<ActiveLearningAlgo> active_learning_algo;
    std::unique_ptr<BasePoolBasedActiveLearningStrategy> strategy;

    if (!strcmp(argv[1], "random")) {
        active_learning_algo = std::move(std::unique_ptr<ActiveLearningAlgo>(
            new PoolBasedPassiveLearningAlgo(
                &model,
                initial_size,
                batch_size,
                max_labels,
                log_filename,
                &get_metric
            )
        ));
    } else {
        if (!strcmp(argv[1], "US")) {
            strategy = std::move(std::unique_ptr<BasePoolBasedActiveLearningStrategy>(
                new PoolBasedUncertaintySamplingStrategy
            ));
        }
        else if (!strcmp(argv[1], "diversity")) {
            strategy = std::move(std::unique_ptr<BasePoolBasedActiveLearningStrategy>(
                new PoolBasedDiversity(0.2)
            ));
        }
        else
            throw 1;

        active_learning_algo = std::move(std::unique_ptr<ActiveLearningAlgo>(
            new PoolBasedActiveLearningAlgo(
                &model,
                strategy.get(),
                initial_size,
                batch_size,
                max_labels,
                log_filename,
                &get_metric
            )
        ));
    }

    test_active_learning_algo(
        active_learning_algo.get(),
        train_pool,
        test_pool,
        permutations,
        start_permutaion_ind,
        permutations_num
    );

    return 0;
}
