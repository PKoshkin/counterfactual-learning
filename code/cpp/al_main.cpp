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
    uint32_t tests_num = 30;
    bool need_help = argc < 2 || !strcmp(argv[1], "-h");
    need_help = need_help || !strcmp(argv[1], "help") || !strcmp(argv[1], "--help");

    if (need_help) {
        std::cout << "Usage:\n./al_main strategy start_permutaion_ind permutations_num";
        std::cout << " batch_size initial_size max_labels\nPossible strategies: ";
        std::cout << "random, US, diversity\n";
        std::cout << "Example:\n./al_main random 0 30 400 5000 9000";
        std::cout << std::endl;
        return 0;
    }

    std::cout << "Running with following params:" << std::endl;
    std::cout << "strategy: " << argv[1] << std::endl;
    uint32_t start_permutaion_ind = get_uint_param(argc, argv, 2, 0, "start_permutaion_ind");
    uint32_t permutations_num = get_uint_param(argc, argv, 3, tests_num, "permutations_num");
    uint32_t batch_size = get_uint_param(argc, argv, 4, 400, "batch_size");
    uint32_t initial_size = get_uint_param(argc, argv, 5, 5000, "initial_size");
    uint32_t max_labels = get_uint_param(argc, argv, 6, 9000, "max_labels");

    uint32_t end_permutaiont_ind = start_permutaion_ind + permutations_num;
    std::string log_filename = "al_test_results.txt";

    char pool_path[100] = "../../pool.json";
    Pool pool = get_pool(pool_path, 90000);
    auto pool_pair = train_test_split(pool, 0.75);
    Pool train_pool = pool_pair.first;
    Pool test_pool = pool_pair.second;

    std::vector<std::vector<int>> permutations = get_permutations(
        tests_num,
        train_pool.size(),
        0
    );

    XGBoostModel base_model(50);
    PositionToFeaturesModel model(&base_model, train_pool.POSITIONS);

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
        &get_metric
    );

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
