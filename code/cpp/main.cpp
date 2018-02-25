#include <iostream>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <cmath>

#include "metric.h"
#include "model.h"
#include "counterfactural_model.h"
#include "xgboost_model.h"


int main() {
    char pool_path[100] = "../../pool.json";
    Pool pool = get_pool(pool_path);

    int tests_num = 50;
    double mean = 0;
    double mean_sq = 0;
    for (long i = 0; i < tests_num; ++i) {
        auto pool_pair = train_test_split(pool, 0.75, (i * 13131 + 12383) % 74017);
        Pool train_pool = pool_pair.first;
        Pool test_pool = pool_pair.second;

        XGBoostModel base_model(100);
        PositionToFeaturesModel model(&base_model, train_pool.POSITIONS);
        model.fit(train_pool);
        std::vector<int> predicted_positions = model.predict(test_pool);
        double metric = get_metric(test_pool, predicted_positions);
        std::cout << "Result metric: " <<  metric << std::endl;
        mean += metric;
        mean_sq += metric * metric;
    }
    mean /= tests_num;
    mean_sq /= tests_num;
    std::cout << "Mean: " << mean << " std: " << std::sqrt(mean_sq - mean * mean) << std::endl;
    return 0;
}
