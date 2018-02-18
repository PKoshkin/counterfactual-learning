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
        const std::vector<std::vector<int>>& permutations,
        uint16_t start_permutaion_ind,
        uint16_t permutations_num,
        std::string file_name,
        uint16_t positions_to_print_number) {
    std::ofstream stream;
    stream.open(file_name, std::ofstream::out | std::ofstream::app);

    if (start_permutaion_ind == 0)
        stream << "Apply " << active_learning_algo->name() << "\n" << std::endl;

    std::vector<double> result_metrics;
    if (permutations_num == permutations.size())
        result_metrics.resize(permutations_num);

    uint16_t end_permutaiont_ind = std::min(
        start_permutaion_ind + permutations_num,
        static_cast<int>(permutations.size())
    );
    for (int j = start_permutaion_ind; j < end_permutaiont_ind; ++j) {
        CounterfacturalModel* model = active_learning_algo->train(train_pool, permutations[j]);

        std::vector<int> predicted_positions = model->predict(test_pool);
        for (int i = 0; i < positions_to_print_number; ++i) {
            stream << "Predicted pos: " << predicted_positions[i] << " Real pos: " << test_pool.positions[i];
            stream << std::endl;
        }

        double metric = get_metric(test_pool, predicted_positions);
        if (permutations_num == permutations.size())
            result_metrics[j] = metric;

        if (positions_to_print_number > 0)
            stream << "Result metric: " << metric << std::endl;
        else
            stream << metric << std::endl;
    }

    if (permutations_num == permutations.size() && permutations_num > 1) {
        double mean = 0, variance = 0;

        for (auto metric: result_metrics)
            mean += metric;
        mean /= permutations_num;

        for (auto metric: result_metrics)
            variance += (metric - mean) * (metric - mean);
        variance /= permutations_num - 1;

        stream << "\nMean: " << mean << " std: " << std::sqrt(variance);
    }

    if (permutations_num + start_permutaion_ind >= permutations.size())
        stream << "\n=============================\n" << std::endl;
    stream.close();
}
