#include <limits>
#include <cmath>

#include "counterfactural_model.h"


CounterfacturalModel::CounterfacturalModel(std::vector<BaseModel*> models) : models(models) {}


void CounterfacturalModel::fit(const Pool& train_pool) {
    std::vector<Pool> splited_pool = train_pool.split_by_positions();
    for (int i = 0; i < splited_pool.size(); ++i) {
        models[i]->fit(splited_pool[i].factors, splited_pool[i].metrics);
    }
}


std::vector<int> CounterfacturalModel::predict(const Pool& test_pool) const {
    std::vector<int> result(test_pool.size());

    for (int i = 0; i < test_pool.size(); ++i) {
        double max_score = -std::numeric_limits<double>::max();
        int max_position = -1;
        for (int model_index = 0; model_index < models.size(); ++model_index) {
            double score = models[model_index]->predict({test_pool.factors[i]})[0];
            if (score > max_score) {
                max_position = test_pool.POSITIONS[model_index];
                max_score = score;
            }
        }
        result[i] = max_position;
    }

    return result;
}


std::vector<std::vector<double>> CounterfacturalModel::predict_proba(const Pool& test_pool) const {
    std::vector<std::vector<double>> result(test_pool.size(), std::vector<double>(models.size()));

    for (int i = 0; i < test_pool.size(); ++i)
        result[i] = predict_proba(test_pool.get(i));

    return result;
}


std::vector<double> CounterfacturalModel::predict_proba(const Object& object) const {
    std::vector<double> result(models.size());
    double exp_sum = 0;

    for (int model_index = 0; model_index < models.size(); ++model_index) {
        result[model_index] = std::exp(models[model_index]->predict({object.factors})[0]);
        exp_sum += result[model_index];
    }

    for (auto& it: result)
        it /= exp_sum;

    return result;
}
