#include <iostream>
#include <ctime>

#include "metric.h"
#include "xgboost_model.h"
#include "counterfactural_model.h"
#include "active_learning_algo.h"
#include "al_utils.h"


int main() {
    uint16_t test_results_sample_size = 30;
    uint16_t start_permutaion_ind = 0;
    uint16_t permutations_num = 1;
    uint16_t end_permutaiont_ind = start_permutaion_ind + permutations_num;
    uint16_t initial_size = 5000;
    uint16_t batch_size = 400;
    uint16_t max_labels = 9000;

    char pool_path[100] = "../../pool.json";
    Pool pool = get_pool(pool_path, 90000);
    auto pool_pair = train_test_split(pool, 0.75);
    Pool train_pool = pool_pair.first;
    Pool test_pool = pool_pair.second;

    std::vector<std::vector<int>> all_permutations = get_permutations(
        test_results_sample_size,
        train_pool.size(),
        0
    );
    std::vector<std::vector<int>> curr_permutations;
    curr_permutations.assign(
        all_permutations.begin() + start_permutaion_ind,
        all_permutations.begin() + std::min(end_permutaiont_ind, test_results_sample_size)
    );

    XGBoostModel base_model(50);
    PositionToFeaturesModel model(&base_model, train_pool.POSITIONS);

    PoolBasedUncertaintySamplingStrategy strategy;
    PoolBasedActiveLearningAlgo active_learning_algo(&model, &strategy, initial_size, batch_size, max_labels);
    // PoolBasedPassiveLearningAlgo active_learning_algo(&model, max_labels);

    test_active_learning_algo(
        &active_learning_algo,
        train_pool,
        test_pool,
        curr_permutations,
        "al_test_results.txt",
        0
    );

    return 0;
}
