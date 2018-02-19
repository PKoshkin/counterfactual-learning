#include "al_utils.h"
#include "metric.h"
#include "model.h"

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
        CounterfacturalModel* model = active_learning_algo->train(train_pool, permutations[j], test_pool);
}
