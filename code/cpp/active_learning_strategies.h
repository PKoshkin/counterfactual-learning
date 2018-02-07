#pragma once


#include "pool.h"
#include "counterfactural_model.h"

#include <string>
#include <list>


class BasePoolBasedActiveLearningStrategy {
public:
    virtual std::string name() = 0;
    virtual double get_score(
        CounterfacturalModel* current_model,
        const std::list<Object>& unlabeled_pool,
        const Pool& labeled_pool,
        const Object& obj) = 0;
};


class PoolBasedUncertaintySamplingStrategy : public BasePoolBasedActiveLearningStrategy{
public:
    virtual std::string name();
    virtual double get_score(
        CounterfacturalModel* current_model,
        const std::list<Object>& unlabeled_pool,
        const Pool& labeled_pool,
        const Object& obj);
};
