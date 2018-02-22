#include "active_learning_algo.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unistd.h>


void ActiveLearningAlgo::start_log(
    std::ofstream& stream,
    const Pool& train_pool,
    const Pool& test_pool
) const {
    if (log_file.size() > 0) {
        stream.open(log_file, std::ofstream::out | std::ofstream::app);
        stream << "\nAlgorithm with following features was applied:\n" << name() << std::endl;
        stream << "train pool size: " << train_pool.size() << std::endl;
        stream << "test pool size: " << test_pool.size() << std::endl;
        stream << std::endl;
    }
}


void BasePoolBasedActiveLearningAlgo::train(
    const Pool& train_pool,
    const std::vector<int>& permutation,
    const Pool& test_pool
) {
    std::ofstream stream;
    start_log(stream, train_pool, test_pool);

    uint32_t current_max_labels = max_labels;
    if (max_labels == 0 || max_labels > train_pool.size())
        current_max_labels = train_pool.size();

    labeled_pool.clear();
    labeled_pool.assign(train_pool, permutation.begin(), permutation.begin() + initial_size);
    labeled_pool.reserve(current_max_labels);

    initialize_train(train_pool, permutation);
    std::cout << "\nStart train algo" << std::endl;
    bool was_error = false;
    for (
        int batch_start = initial_size;
        batch_start < max_labels;
        batch_start += batch_size
    ) {
        for (int try_index = 0; try_index < tries_number; ++try_index) {
            try {
                make_iteration(
                    train_pool,
                    test_pool,
                    permutation,
                    batch_start,
                    current_max_labels,
                    stream
                );
                break;
            }
            catch(...) {
                std::cout << "An error occured on " << try_index + 1 << " attempt" << std::endl;
                sleep(3);
            }
            if (try_index == tries_number - 1)
                was_error = true;
        }
        if (was_error)
            break;
    }

    if (log_file.size() > 0) {
        if (was_error)
            stream << "error!!!" << std::endl;
        stream << std::endl;
        stream.close();
    }
    std::cout << "\nEnd train algo" << std::endl;
}



PoolBasedActiveLearningAlgo::PoolBasedActiveLearningAlgo(
    CounterfacturalModel* model,
    BasePoolBasedActiveLearningStrategy* strategy,
    uint32_t initial_size,
    uint32_t batch_size,
    uint32_t max_labels,
    std::string log_file,
    Metric* metric
) : strategy(strategy) {
    this->batch_size = batch_size;
    this->max_labels = max_labels;
    this->model = model;
    this->metric = metric;
    this->log_file = log_file;
    this->initial_size = initial_size;
}


std::string PoolBasedActiveLearningAlgo::name() const {
    std::string result = "";

    result += "algo: pool-based\n";
    result += "strategy: " + strategy->name() + "\n";
    result += "initial size: " + std::to_string(initial_size) + "\n";
    result += "batch size: " + std::to_string(batch_size) + "\n";
    result += "max queries: " + std::to_string(max_labels);

    return result;
}


void PoolBasedActiveLearningAlgo::initialize_train(
    const Pool& train_pool,
    const std::vector<int>& permutation
) {
    unlabeled_indexes.clear();
    for (uint32_t index = initial_size; index < permutation.size(); ++index)
        unlabeled_indexes.push_back(permutation[index]);
    if (!(strategy->is_model_free()))
        model->fit(labeled_pool);

    strategy->initialize(train_pool, permutation, initial_size);
}


void PoolBasedActiveLearningAlgo::make_iteration(
    const Pool& train_pool,
    const Pool& test_pool,
    const std::vector<int>& permutation,
    uint32_t batch_start,
    uint32_t current_max_labels,
    std::ofstream& stream
) {
    std::list<std::pair<std::list<int>::iterator, double>> batch;
    uint32_t curr_batch_size = std::min(
        batch_size,
        uint32_t(current_max_labels - batch_start)
    );
    for (
        auto unlabeled_iter = unlabeled_indexes.begin();
        unlabeled_iter != unlabeled_indexes.end();
        ++unlabeled_iter
    ) {
        double score = strategy->get_score(model, train_pool, *unlabeled_iter);
        bool suit = false;

        for (auto it = batch.begin(); it != batch.end(); ++it)
            if (it->second < score) {
                batch.insert(it, {unlabeled_iter, score});
                suit = true;
                break;
            }

        if (suit && batch.size() > curr_batch_size)
            batch.pop_back();
        if (!suit && batch.size() < curr_batch_size)
            batch.push_back({unlabeled_iter, score});
    }
    std::vector<int> batch_ind;
    batch_ind.reserve(batch.size());
    for (auto it: batch) {
        batch_ind.push_back(*(it.first));
        labeled_pool.push_back(train_pool.get(*(it.first)));
        unlabeled_indexes.erase(it.first);
    }

    strategy->update(train_pool, batch_ind, unlabeled_indexes);

    if (log_file.size() > 0 || !strategy->is_model_free())
        model->fit(labeled_pool);
    if (log_file.size() > 0) {
        std::vector<int> predictions = model->predict(test_pool);
        stream << metric(test_pool, predictions) << std::endl;
    }
}


void PoolBasedPassiveLearningAlgo::initialize_train(
    const Pool& train_pool,
    const std::vector<int>& permutation
) {}


void PoolBasedPassiveLearningAlgo::make_iteration(
    const Pool& train_pool,
    const Pool& test_pool,
    const std::vector<int>& permutation,
    uint32_t batch_start,
    uint32_t current_max_labels,
    std::ofstream& stream
) {
    int batch_end = std::min(batch_start + batch_size, current_max_labels);
    for (int obj_ind = batch_start; obj_ind < batch_end; ++obj_ind)
        labeled_pool.push_back(train_pool.get(permutation[obj_ind]));
    model->fit(labeled_pool);
    if (log_file.size() > 0) {
        std::vector<int> predictions = model->predict(test_pool);
        stream << metric(test_pool, predictions) << std::endl;
    }
}


std::string PoolBasedPassiveLearningAlgo::name() const {
    std::string result = "";
    result += "algo: pool-based\n";
    result += "strategy: random\n";
    result += "initial size: " + std::to_string(initial_size) + "\n";
    result += "batch size: " + std::to_string(batch_size) + "\n";
    result += "max queries: " + std::to_string(max_labels);

    return result;
}


PoolBasedPassiveLearningAlgo::PoolBasedPassiveLearningAlgo(
    CounterfacturalModel* model,
    uint32_t initial_size,
    uint32_t batch_size,
    uint32_t max_labels,
    std::string log_file,
    Metric* metric
) {
    this->batch_size = batch_size;
    this->max_labels = max_labels;
    this->model = model;
    this->metric = metric;
    this->log_file = log_file;
    this->initial_size = initial_size;
}
