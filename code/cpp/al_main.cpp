#include <iostream>
#include <ctime>

#include "metric.h"
#include "xgboost_model.h"
#include "counterfactural_model.h"
#include "active_learning_algo.h"
#include "al_utils.h"


int main() {
    std::srand(std::time(0));
    char pool_path[100] = "../../pool.json";
    Pool pool = get_pool(pool_path, 90000);
    auto pool_pair = train_test_split(pool, 0.75);
    Pool train_pool = pool_pair.first;
    Pool test_pool = pool_pair.second;

    // std::vector<LogisticRegression> models_vector(train_pool.POSITIONS.size(), LogisticRegression(0.0000005, 1, 16, 5, 100));
    std::vector<XGBoostModel> models_vector(train_pool.POSITIONS.size(), XGBoostModel(10));
    std::vector<BaseModel*> models_pointers(models_vector.size());
    for (int i = 0; i < models_vector.size(); ++i)
        models_pointers[i] = &models_vector[i];

    ElevenRegressions model(models_pointers);
    PoolBasedUncertaintySamplingStrategy strategy;
    PoolBasedActiveLearningAlgo active_learning_algo(&model, &strategy, 1000, 200, 2000);
    test_active_learning_algo(&active_learning_algo, train_pool, test_pool, "al_test_results.txt", 2);

    return 0;
}
