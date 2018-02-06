#include "al_utils.h"
#include "metric.h"
#include "model.h"

#include <fstream>
#include <iostream>
#include <cmath>


void test_pool_based_strategy(
        BasePoolBasedActiveLearningStrategy* strategy,
        const Pool& train_pool,
        const Pool& test_pool,
        std::string file_name,
        uint16_t iteration_number,
        uint16_t pos_to_print_number) {
    std::ofstream stream;
    stream.open(file_name);

    //stream << "Apply " << PoolBasedActiveLearningAlgo.name << " active learning algorithm";
    //stream << " using " << strategy->name << " strategy\n" << std::endl;

    std::vector<double> result_metrics(iteration_number);
    double mean = 0, variance = 0;

    for (int j = 0; j < iteration_number; ++j) {
        std::vector<LogisticRegression> models_vector(train_pool.POSITIONS.size(), LogisticRegression(0.0000005, 1, 16, 5, 100));
        std::vector<BaseModel*> models_pointers(models_vector.size());
        for (int i = 0; i < models_vector.size(); ++i)
            models_pointers[i] = &models_vector[i];

        CounterfacturalModel model(models_pointers);
        PoolBasedActiveLearningAlgo active_learning_algo(model, train_pool, strategy, 1000, 200, 2000);
        model = active_learning_algo.train();

        std::vector<int> predicted_positions = model.predict(test_pool);

        for (int i = 0; i < pos_to_print_number; ++i)
            stream << "Predicted pos: " << predicted_positions[i] << " Real pos: " << test_pool.positions[i] << std::endl;

        result_metrics[j] = get_metric(test_pool, predicted_positions);

        stream << "Result metric: " << result_metrics[j] << std::endl;

        mean += result_metrics[j];
    }

    mean /= iteration_number;
    for (auto metric: result_metrics)
        variance += (metric - mean) * (metric - mean);

    variance /= iteration_number - 1;

    stream << "\nMean: " << mean << " std: " << std::sqrt(variance) << "\n=======================\n" << std::endl;

    stream.close();
}
