#include "test_model.h"
#include "utils.h"
#include "metric.h"

#include <string>


SimpleClassification::SimpleClassification(uint16_t num_iteration, uint16_t num_class)
    : model(num_iteration, {{"objective", "multi:softprob"}, {"num_class", std::to_string(num_class)}}),
    num_class(num_class) {
}


void SimpleClassification::fit(const Pool& train_pool) {
    model.fit(train_pool.factors, train_pool.metrics);
}


std::vector<int> SimpleClassification::predict(const Pool& test_pool) const {
    std::vector<int> result(test_pool.size());

    for (int i = 0; i < test_pool.size(); ++i) {
        std::vector<double> probas = model.predict_proba(test_pool.factors[i]);
        uint16_t argmax;
        double max = -1;

        for (uint16_t answer_ind = 0; answer_ind < num_class; ++answer_ind)
            if (probas[answer_ind] > max) {
                argmax = answer_ind;
                max = probas[answer_ind];
            }

        result[i] = argmax;
    }

    return result;
}


std::vector<double> SimpleClassification::predict_proba(const Object& object) const {
    return model.predict_proba(object.factors);
}
