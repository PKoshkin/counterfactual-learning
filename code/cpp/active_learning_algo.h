#pragma once


#include "counterfactural_model.h"
#include "active_learning_strategies.h"
#include "metric.h"

#include <list>
#include <string>


class ActiveLearningAlgo {
protected:
    CounterfacturalModel* model;
    uint16_t initial_size;
    Metric* metric;
    std::string log_file;
public:
    virtual std::string name() const = 0;
    virtual CounterfacturalModel* train(
        const Pool& train_pool,
        const std::vector<int>& permutation,
        const Pool& test_pool
    ) = 0;
};


class PoolBasedActiveLearningAlgo : public ActiveLearningAlgo {
private:
    uint16_t max_labels;
    uint16_t batch_size;
    BasePoolBasedActiveLearningStrategy* strategy;
public:
    virtual std::string name() const;
    PoolBasedActiveLearningAlgo(
        CounterfacturalModel* model,
        BasePoolBasedActiveLearningStrategy* strategy,
        uint16_t initial_size,
        uint16_t batch_size,
        uint16_t max_labels,
        std::string log_file,
        Metric* metric
    );
    virtual CounterfacturalModel* train(
        const Pool& train_pool,
        const std::vector<int>& permutation,
        const Pool& test_pool
    );
};


class PoolBasedPassiveLearningAlgo : public ActiveLearningAlgo {
private:
    uint16_t max_labels;
    uint16_t batch_size;
public:
    virtual std::string name() const;
    PoolBasedPassiveLearningAlgo(
        CounterfacturalModel* model,
        uint16_t initial_size,
        uint16_t batch_size,
        uint16_t max_labels,
        std::string log_file,
        Metric* metric
    );
    virtual CounterfacturalModel* train(
        const Pool& train_pool,
        const std::vector<int>& permutation,
        const Pool& test_pool
    );
};
