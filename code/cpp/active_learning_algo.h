#pragma once


#include "counterfactural_model.h"
#include "active_learning_strategies.h"
#include "metric.h"

#include <list>
#include <string>


class ActiveLearningAlgo {
protected:
    CounterfacturalModel* model;
    uint32_t initial_size;
    Metric* metric;
    std::string log_file;
    void start_log(
        std::ofstream& stream,
        const Pool& train_pool,
        const Pool& test_pool
    ) const;
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
    uint32_t max_labels;
    uint32_t batch_size;
    BasePoolBasedActiveLearningStrategy* strategy;
public:
    virtual std::string name() const;
    PoolBasedActiveLearningAlgo(
        CounterfacturalModel* model,
        BasePoolBasedActiveLearningStrategy* strategy,
        uint32_t initial_size,
        uint32_t batch_size,
        uint32_t max_labels,
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
    uint32_t max_labels;
    uint32_t batch_size;
public:
    virtual std::string name() const;
    PoolBasedPassiveLearningAlgo(
        CounterfacturalModel* model,
        uint32_t initial_size,
        uint32_t batch_size,
        uint32_t max_labels,
        std::string log_file,
        Metric* metric
    );
    virtual CounterfacturalModel* train(
        const Pool& train_pool,
        const std::vector<int>& permutation,
        const Pool& test_pool
    );
};
