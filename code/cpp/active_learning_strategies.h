#pragma once


#include "utils.h"
#include "counterfactural_model.h"


template<class Model>
class BasePoolBasedActiveLearningStrategy {
public:
    virtual double get_score(const CounterfacturalModel<Model>& current_model,
                             const Pool& unlabeled_pool,
                             const Pool& labeled_pool,
                             uint16_t index) = 0;
};


template<class Model>
class PoolBasedUncertaintySamplingStrategy : public BasePoolBasedActiveLearningStrategy<Model>{
    virtual double get_score(const CounterfacturalModel<Model>& current_model,
                             const Pool& unlabeled_pool,
                             const Pool& labeled_pool,
                             uint16_t index);
};


template<class Model>
double PoolBasedUncertaintySamplingStrategy<Model>::get_score(const CounterfacturalModel<Model>& current_model,
                                                               const Pool& unlabeled_pool,
                                                               const Pool& labeled_pool,
                                                               uint16_t index) {
    std::vector<double> probas = current_model.predict_proba(unlabeled_pool.get(index));
    double gini_score = 1;
    for (auto proba: probas)
        gini_score -= proba * proba;

    return gini_score;
}

