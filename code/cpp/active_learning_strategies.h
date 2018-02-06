#pragma once


#include "utils.h"
#include "counterfactural_model.h"

#include <string>
#include <list>


template<class Model>
class BasePoolBasedActiveLearningStrategy {
public:
    virtual double get_score(const CounterfacturalModel<Model>& current_model,
                             const std::list<Object>& unlabeled_pool,
                             const Pool& labeled_pool,
                             const Object& obj) = 0;
};


template<class Model>
class PoolBasedUncertaintySamplingStrategy : public BasePoolBasedActiveLearningStrategy<Model>{
public:
    static std::string name = "Uncertainty Sampling";
    virtual double get_score(const CounterfacturalModel<Model>& current_model,
                             const std::list<Object>& unlabeled_pool,
                             const Pool& labeled_pool,
                             const Object& obj);
};


template<class Model>
double PoolBasedUncertaintySamplingStrategy<Model>::get_score(const CounterfacturalModel<Model>& current_model,
                                                              const std::list<Object>& unlabeled_pool,
                                                              const Pool& labeled_pool,
                                                              const Object& obj) {
    std::vector<double> probas = current_model.predict_proba(obj);

    double gini_score = 1;
    for (auto proba: probas)
        gini_score -= proba * proba;

    return gini_score;
}

