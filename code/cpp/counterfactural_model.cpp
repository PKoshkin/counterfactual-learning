#include <limits>

#include "counterfactural_model.h"


std::vector<std::vector<double>> CounterfacturalModel::predict_proba(const Pool& test_pool) const {
    std::vector<std::vector<double>> result(test_pool.size(), std::vector<double>(test_pool.POSITIONS.size()));

    for (int i = 0; i < test_pool.size(); ++i)
        result[i] = predict_proba(test_pool.get(i));

    return result;
}


ElevenRegressionsModel::ElevenRegressionsModel(std::vector<BaseModel*> models) : models(models) {}


void ElevenRegressionsModel::fit(const Pool& train_pool) {
    std::vector<Pool> splited_pool = train_pool.split_by_positions();
    for (int i = 0; i < splited_pool.size(); ++i) {
        models[i]->fit(splited_pool[i].factors, splited_pool[i].metrics);
    }
}


std::vector<int> ElevenRegressionsModel::predict(const Pool& test_pool) const {
    std::vector<int> result(test_pool.size());

    for (int i = 0; i < test_pool.size(); ++i) {
        double max_score = -std::numeric_limits<double>::max();
        int max_position = -1;
        for (int model_index = 0; model_index < models.size(); ++model_index) {
            double score = models[model_index]->predict(test_pool.factors[i]);
            if (score > max_score) {
                max_position = test_pool.POSITIONS[model_index];
                max_score = score;
            }
        }
        result[i] = max_position;
    }

    return result;
}


std::vector<double> ElevenRegressionsModel::predict_proba(const Object& object) const {
    std::vector<double> result(models.size());

    for (int model_index = 0; model_index < models.size(); ++model_index)
        result[model_index] = models[model_index]->predict(object.factors);

    softmax(result);
    return result;
}


PositionToFeaturesModel::PositionToFeaturesModel(BaseModel* model, const std::vector<int>& positions)
    : model(model), positions(positions) {}


void PositionToFeaturesModel::fit(const Pool& train_pool) {
    Matrix features = train_pool.factors;
    for (int i = 0; i < features.size(); ++i)
        features[i].push_back(static_cast<double>(train_pool.positions[i]));

    model->fit(features, train_pool.metrics);
}


std::vector<int> PositionToFeaturesModel::predict(const Pool& test_pool) const {
    std::vector<int> result(test_pool.size());

    for (int i = 0; i < test_pool.size(); ++i) {
        std::vector<double> features = test_pool.factors[i];
        features.resize(features.size() + 1);
        double max_score = -std::numeric_limits<double>::max();
        int max_position = -1;
        for (auto pos: test_pool.POSITIONS) {
            features[features.size() - 1] = pos;
            double score = model->predict(features);
            if (score > max_score) {
                max_position = pos;
                max_score = score;
            }
        }

        result[i] = max_position;
    }

    return result;
}


std::vector<double> PositionToFeaturesModel::predict_proba(const Object& object) const {
    std::vector<double> result(positions.size());

    for (int i = 0; i < result.size(); ++i) {
        std::vector<double> features = object.factors;
        features.push_back(static_cast<double>(object.position));
        result[i] = model->predict(features);
    }

    softmax(result);
    return result;
}
