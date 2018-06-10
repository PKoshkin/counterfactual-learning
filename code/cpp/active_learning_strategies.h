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
        const std::vector<int>& permutation,
        int labeled_pool_size
    ) = 0;
    virtual void update(
        const Pool& train_pool,
        const std::vector<int>& batch,
        const std::list<int>& unlabeled_indexes
    ) = 0;
    virtual double get_score(
        CounterfacturalModel* current_model,
        const Pool& train_pool,
        int index
    ) = 0;
};


class PoolBasedUncertaintySamplingStrategy : public BasePoolBasedActiveLearningStrategy {
public:
    virtual bool is_model_free();
    virtual std::string name();
    virtual void initialize(
        const Pool& train_pool,
        const std::vector<int>& permutation,
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
    std::vector<double> object_norms;

    double cosin_similarity(const Matrix& factors, uint32_t ind_a, uint32_t ind_b);
public:
    PoolBasedDiversity(double seen_labeled_objects_share);
    virtual bool is_model_free();
    virtual std::string name();
    virtual void initialize(
        const Pool& train_pool,
        const std::vector<int>& permutation,
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

/*
class PoolBasedQBC : public BasePoolBasedActiveLearningStrategy {
    std::vector<CounterfacturalModel*> committee;
    std::vector<Pool> train_pools;
    uint32_t iteration_ind;
    std::vector<std::vector<std::vector<<double>>> committee_probas;
public:
    PoolBasedQBC(std::vector<CounterfacturalModel*>&& committee);
    virtual bool is_model_free();
    virtual std::string name();
    virtual void initialize(
        const Pool& train_pool,
        const std::vector<int>& permutation,
        int labeled_pool_size
    );
    virtual void update(
        const Pool& train_pool,
        const std::vector<int>& batch,
        const std::list<int>& unlabeled_indexes
    );
    virtual double get_score(
        CounterfacturalModel* current_model,
        const Pool& train_pool,
        int index
    );
}
*/
