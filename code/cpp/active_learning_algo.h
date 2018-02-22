#pragma once


#include "counterfactural_model.h"
#include "active_learning_strategies.h"
#include "metric.h"

#include <list>
#include <string>


class ActiveLearningAlgo {
protected:
    uint16_t tries_number = 3;
    CounterfacturalModel* model;
    uint32_t initial_size;
    Metric* metric;
    std::string log_file;
    Pool labeled_pool;

    void start_log(
        std::ofstream& stream,
        const Pool& train_pool,
        const Pool& test_pool
    ) const;
public:
    virtual std::string name() const = 0;
    virtual void train(
        const Pool& train_pool,
        const std::vector<int>& permutation,
        const Pool& test_pool
    ) = 0;
};


class BasePoolBasedActiveLearningAlgo : public ActiveLearningAlgo {
protected:
    uint32_t max_labels;
    uint32_t batch_size;
public:
    virtual std::string name() const = 0;
    virtual void train(
        const Pool& train_pool,
        const std::vector<int>& permutation,
        const Pool& test_pool
    );
    virtual void initialize_train(
        const Pool& train_pool,
        const std::vector<int>& permutation
    ) = 0;
    virtual void make_iteration(
        const Pool& train_pool,
        const Pool& test_pool,
        const std::vector<int>& permutation,
        uint32_t batch_start,
        uint32_t current_max_labels,
        std::ofstream& stream
    ) = 0;
};


class PoolBasedActiveLearningAlgo : public BasePoolBasedActiveLearningAlgo {
private:
    BasePoolBasedActiveLearningStrategy* strategy;
    std::list<int> unlabeled_indexes;
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
    virtual void initialize_train(
        const Pool& train_pool,
        const std::vector<int>& permutation
    );
    virtual void make_iteration(
        const Pool& train_pool,
        const Pool& test_pool,
        const std::vector<int>& permutation,
        uint32_t batch_start,
        uint32_t current_max_labels,
        std::ofstream& stream
    );
};


class PoolBasedPassiveLearningAlgo : public BasePoolBasedActiveLearningAlgo {
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
    virtual void initialize_train(
        const Pool& train_pool,
        const std::vector<int>& permutation
    );
    virtual void make_iteration(
        const Pool& train_pool,
        const Pool& test_pool,
        const std::vector<int>& permutation,
        uint32_t batch_start,
        uint32_t current_max_labels,
        std::ofstream& stream
    );
};
