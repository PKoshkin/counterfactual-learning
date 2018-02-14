#pragma once


#include "pool.h"
#include "counterfactural_model.h"

#include <string>
#include <list>


class BasePoolBasedActiveLearningStrategy {
public:
    virtual bool is_model_free() = 0;
    virtual std::string name() = 0;
    virtual void initialize(
        const Pool& train_pool,
        const std::vector<int>& indexes_permutation,
        int labeled_pool_size) = 0;
    virtual void update(
        const Pool& train_pool,
        const std::vector<int>& batch,
        const std::list<int>& unlabeled_indexes) = 0;
    virtual double get_score(
        CounterfacturalModel* current_model,
        const Pool& train_pool,
        int index) = 0;
};


class PoolBasedUncertaintySamplingStrategy : public BasePoolBasedActiveLearningStrategy {
public:
    virtual bool is_model_free();
    virtual std::string name();
    virtual void initialize(
        const Pool& train_pool,
        const std::vector<int>& indexes_permutation,
        int labeled_pool_size);
    virtual void update(
        const Pool& train_pool,
        const std::vector<int>& batch,
        const std::list<int>& unlabeled_indexes);
    virtual double get_score(
        CounterfacturalModel* current_model,
        const Pool& train_pool,
        int index);
};


class PoolBasedDiversity : public BasePoolBasedActiveLearningStrategy {
private:
    double seen_labeled_objects_share;
    std::vector<double> current_scores;
public:
    PoolBasedDiversity(double seen_labeled_objects_share);
    virtual bool is_model_free();
    virtual std::string name();
    virtual void initialize(
        const Pool& train_pool,
        const std::vector<int>& indexes_permutation,
        int labeled_pool_size);
    virtual void update(
        const Pool& train_pool,
        const std::vector<int>& batch,
        const std::list<int>& unlabeled_indexes);
    virtual double get_score(
        CounterfacturalModel* current_model,
        const Pool& train_pool,
        int index);
};
