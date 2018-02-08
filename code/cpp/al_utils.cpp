#include "al_utils.h"
#include "metric.h"
#include "model.h"

#include <fstream>
#include <iostream>
#include <cmath>


void test_active_learning_algo(
        ActiveLearningAlgo* active_learning_algo,
        const Pool& train_pool,
        const Pool& test_pool,
        std::string file_name,
        uint16_t iteration_number,
        uint16_t positions_to_print_number) {
    std::ofstream stream;
    stream.open(file_name);

    stream << "Apply " << active_learning_algo->name() << "\n" << std::endl;

    std::vector<double> result_metrics(iteration_number);

    for (int j = 0; j < iteration_number; ++j) {
        CounterfacturalModel* model = active_learning_algo->train(train_pool);

        std::vector<int> predicted_positions = model->predict(test_pool);
        for (int i = 0; i < positions_to_print_number; ++i)
            stream << "Predicted pos: " << predicted_positions[i] << " Real pos: " << test_pool.positions[i] << std::endl;

        result_metrics[j] = get_metric(test_pool, predicted_positions);
        stream << "Result metric: " << result_metrics[j] << std::endl;
    }

    if (iteration_number > 1) {
        double mean = 0, variance = 0;

        for (auto metric: result_metrics)
            mean += metric;
        mean /= iteration_number;

        for (auto metric: result_metrics)
            variance += (metric - mean) * (metric - mean);
        variance /= iteration_number - 1;

        stream << "\nMean: " << mean << " std: " << std::sqrt(variance);
    }

    stream << "\n=============================\n" << std::endl;
    stream.close();
}
