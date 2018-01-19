#include <limits>

#include "utils.h"


template<class Model>
class CounterfacturalModel {
private:
    std::vector<Model> models;
public:
    CounterfacturalModel(std::vector<Model> models);
    void fit(const Pool& train_pool);
    std::vector<int> predict(const Pool& test_pool);
};


template<class Model>
CounterfacturalModel<Model>::CounterfacturalModel(std::vector<Model> models) : models(models) {}


template<class Model>
void CounterfacturalModel<Model>::fit(const Pool& train_pool) {
    std::vector<Pool> splited_pool = train_pool.split_by_positions();
    for (int i = 0; i < splited_pool.size(); ++i) {
        std::cout << std::endl << i << " model start training\n" << std::endl;
        models[i].fit(splited_pool[i].factors, splited_pool[i].metrics);
    }
}


template<class Model>
std::vector<int> CounterfacturalModel<Model>::predict(const Pool& test_pool) {
    std::vector<int> result(test_pool.size());

    for (int i = 0; i < test_pool.size(); ++i) {
        double max_score = -std::numeric_limits<double>::max();
        int max_position = -1;
        for (int model_index = 0; model_index < models.size(); ++model_index) {
            double score = models[model_index].predict({test_pool.factors[i]})[0];
            if (score > max_score) {
                max_position = test_pool.POSITIONS[model_index];
                max_score = score;
            }
        }
        result[i] = max_position;
    }

    return result;
}
