#pragma once


#include "counterfactural_model.h"
#include "active_learning_strategies.h"

#include <list>


class ActiveLearningAlgo {
protected:
    CounterfacturalModel* model;
    uint16_t initial_size;
public:
    virtual std::string name() = 0;
    virtual CounterfacturalModel* train(const Pool& train_pool, const std::vector<int>& permutation) = 0;
};


class PoolBasedActiveLearningAlgo : public ActiveLearningAlgo {
private:
    uint16_t max_labels;
    uint16_t batch_size;
    BasePoolBasedActiveLearningStrategy* strategy;
public:
    virtual std::string name();
    PoolBasedActiveLearningAlgo(
            CounterfacturalModel* model,
            BasePoolBasedActiveLearningStrategy* strategy,
            uint16_t initial_size = 1000,
            uint16_t batch_size = 200,
            uint16_t max_labels = 2000
    );
    virtual CounterfacturalModel* train(const Pool& train_pool, const std::vector<int>& permutation);
};


class PoolBasedPassiveLearningAlgo : public ActiveLearningAlgo {
private:
    uint16_t max_labels;
public:
    virtual std::string name();
    PoolBasedPassiveLearningAlgo(
            CounterfacturalModel* model,
            uint16_t max_labels = 2000
    );
    virtual CounterfacturalModel* train(const Pool& train_pool, const std::vector<int>& permutation);
};
