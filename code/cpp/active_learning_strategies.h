#pragma once


#include "pool.h"
#include "counterfactural_model.h"

#include <string>
#include <list>


class BasePoolBasedActiveLearningStrategy {
public:
    virtual bool is_model_free() = 0;
    virtual std::string name() = 0;
    virtual double get_score(
        CounterfacturalModel* current_model,
        const std::list<Object>& unlabeled_pool,
        const Pool& labeled_pool,
        const Object& obj) = 0;
};


class PoolBasedUncertaintySamplingStrategy : public BasePoolBasedActiveLearningStrategy {
public:
    virtual bool is_model_free();
    virtual std::string name();
    virtual double get_score(
        CounterfacturalModel* current_model,
        const std::list<Object>& unlabeled_pool,
        const Pool& labeled_pool,
        const Object& obj);
};


class PoolBasedDiversity : public BasePoolBasedActiveLearningStrategy {
public:
    virtual bool is_model_free();
    virtual std::string name();
    virtual double get_score(
        CounterfacturalModel* current_model,
        const std::list<Object>& unlabeled_pool,
        const Pool& labeled_pool,
        const Object& obj);
};
