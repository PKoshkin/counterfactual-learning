#include "al_utils.h"
#include "metric.h"
#include "model.h"

#include <cstring>
#include <cmath>


void test_active_learning_algo(
        ActiveLearningAlgo* active_learning_algo,
        const Pool& train_pool,
        const Pool& test_pool,
        const std::vector<std::vector<int>>& permutations,
        uint16_t start_permutaion_ind,
        uint16_t permutations_num) {
    uint16_t end_permutaiont_ind = std::min(
        start_permutaion_ind + permutations_num,
        static_cast<int>(permutations.size())
    );

    for (int j = start_permutaion_ind; j < end_permutaiont_ind; ++j)
        active_learning_algo->train(train_pool, permutations[j], test_pool);
}


void set_algo(
    const char strategy_name[],
    std::unique_ptr<ActiveLearningAlgo>& active_learning_algo,
    std::unique_ptr<BasePoolBasedActiveLearningStrategy>& strategy,
    CounterfacturalModel& model,
    uint32_t initial_size,
    uint32_t batch_size,
    uint32_t max_labels,
    std::string log_filename,
    Metric* metric
) {
    std::string name(strategy_name);
    if (name == "random") {
        active_learning_algo = std::move(std::unique_ptr<ActiveLearningAlgo>(
            new PoolBasedPassiveLearningAlgo(
                &model,
                initial_size,
                batch_size,
                max_labels,
                log_filename,
                metric
            )
        ));
    } else {
        if (name == "US") {
            strategy = std::move(std::unique_ptr<BasePoolBasedActiveLearningStrategy>(
                new PoolBasedUncertaintySamplingStrategy
            ));
        }
        else if (name.substr(0, 9) == "diversity") {
            double share = 0;
            if (name.size() > 9)
                share = std::stof(name.substr(9));
            strategy = std::move(std::unique_ptr<BasePoolBasedActiveLearningStrategy>(
                new PoolBasedDiversity(share)
            ));
        }
        else {
            std::string mes = std::string("Invalid strategy: ") + std::string(strategy_name);
            throw std::invalid_argument(mes);
        }

        active_learning_algo = std::move(std::unique_ptr<ActiveLearningAlgo>(
            new PoolBasedActiveLearningAlgo(
                &model,
                strategy.get(),
                initial_size,
                batch_size,
                max_labels,
                log_filename,
                metric
            )
        ));
    }
}
