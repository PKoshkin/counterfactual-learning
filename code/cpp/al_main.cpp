#include <iostream>
#include <ctime>

#include "metric.h"
#include "model.h"
#include "counterfactural_model.h"
#include "active_learning_strategies.h"
#include "active_learning_algo.h"


int main() {
    std::srand(std::time(0));
    char pool_path[100] = "../../pool.json";
    Pool pool = get_pool(pool_path, 90000);
    auto pool_pair = train_test_split(pool, 0.75);
    Pool train_pool = pool_pair.first;
    Pool test_pool = pool_pair.second;

    std::vector<LogisticRegression> models_vector(pool.POSITIONS.size(), LogisticRegression(0.0000005, 1, 16, 100, 100));
    CounterfacturalModel<LogisticRegression> model(models_vector);
    PoolBasedUncertaintySamplingStrategy<LogisticRegression> strategy;
    PoolBasedActiveLearningAlgo<LogisticRegression> active_learning_algo(model, pool, &strategy);
    model = active_learning_algo.train();

    std::vector<int> predicted_positions = model.predict(test_pool);

    for (int i = 0; i < 10; ++i)
        std::cout << "Predicted pos: " << predicted_positions[i] << " Real pos: " << test_pool.positions[i] << std::endl;

    std::cout << "Result metric: " << get_metric(test_pool, predicted_positions) << std::endl;
    return 0;
}
