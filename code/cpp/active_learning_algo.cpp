#include "active_learning_algo.h"

#include <algorithm>
#include <fstream>
#include <iostream>


PoolBasedActiveLearningAlgo::PoolBasedActiveLearningAlgo(
        CounterfacturalModel* model,
        BasePoolBasedActiveLearningStrategy* strategy,
        uint16_t initial_size,
        uint16_t batch_size,
        uint16_t max_labels,
        std::string log_file,
        Metric* metric) : strategy(strategy), batch_size(batch_size), max_labels(max_labels) {
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


CounterfacturalModel* PoolBasedActiveLearningAlgo::train(
        const Pool& train_pool,
        const std::vector<int>& permutation,
        const Pool& test_pool) {

    std::ofstream stream;
    if (log_file.size() > 0) {
        stream.open(log_file, std::ofstream::out | std::ofstream::app);
        stream << "Apply " << name() << "\n" << std::endl;
    }

    uint16_t current_max_labels = max_labels;
    if (max_labels == 0 || max_labels > train_pool.size())
        current_max_labels = train_pool.size();

    std::list<int> unlabeled_indexes;
    for (int index = initial_size; index < permutation.size(); ++index)
        unlabeled_indexes.push_back(permutation[index]);

    Pool labeled_pool;
    labeled_pool.assign(train_pool, permutation.begin(), permutation.begin() + initial_size);
    labeled_pool.reserve(current_max_labels);

    if (!(strategy->is_model_free()))
        model->fit(labeled_pool);

    strategy->initialize(train_pool, permutation, initial_size);
    std::cout << "\nStart active learning train" << std::endl;

    while (labeled_pool.size() < current_max_labels) {
        std::list<std::pair<std::list<int>::iterator, double>> batch;
        uint16_t curr_batch_size = std::min(batch_size, uint16_t(current_max_labels - labeled_pool.size()));
        for (auto unlabeled_ind = unlabeled_indexes.begin() ; unlabeled_ind != unlabeled_indexes.end(); ++unlabeled_ind) {
            double score = strategy->get_score(model, train_pool, *unlabeled_ind);
            bool suit = false;

            for (auto it = batch.begin(); it != batch.end(); ++it)
                if (it->second < score) {
                    batch.insert(it, {unlabeled_ind, score});
                    suit = true;
                    break;
                }

            if (suit && batch.size() > curr_batch_size)
                batch.pop_back();
            if (!suit && batch.size() < curr_batch_size)
                batch.push_back({unlabeled_ind, score});
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

        if (strategy->is_model_free())
        model->fit(labeled_pool);

    if (log_file.size() > 0) {
        stream << std::endl;
        stream.close();
    }

    std::cout << "\nEnd active learning train" << std::endl;

    return model;
}


std::string PoolBasedPassiveLearningAlgo::name() const {
    std::string result = "";
    result += "algo: pool-based\n";
    result += "strategy: random\n";
    result += "initial size: " + std::to_string(initial_size) + "\n";
    result += "batch size: " + std::to_string(batch_size) + "\n";
    result += "max queries: " + std::to_string(max_labels) + "\n";

    return result;
}


PoolBasedPassiveLearningAlgo::PoolBasedPassiveLearningAlgo(
        CounterfacturalModel* model,
        uint16_t initial_size,
        uint16_t batch_size,
        uint16_t max_labels,
        std::string log_file,
        Metric* metric) : max_labels(max_labels), batch_size(batch_size) {
    this->model = model;
    this->metric = metric;
    this->log_file = log_file;
    this->initial_size = initial_size;
}


CounterfacturalModel* PoolBasedPassiveLearningAlgo::train(
        const Pool& train_pool,
        const std::vector<int>& permutation,
        const Pool& test_pool) {
    Pool actual_train_pool;
    actual_train_pool.assign(train_pool, permutation.begin(), permutation.begin() + initial_size);
    actual_train_pool.reserve(max_labels);

    std::ofstream stream;
    if (log_file.size() > 0) {
        stream.open(log_file, std::ofstream::out | std::ofstream::app);
        stream << "Algorithm with following features was applied:\n" << name() << "\n" << std::endl;
    }

    for (int batch_start = initial_size; batch_start < max_labels; batch_start += batch_size) {
        int batch_end = std::min(batch_start + batch_size, static_cast<int>(max_labels));
        for (int obj_ind = batch_start; obj_ind < batch_end; ++obj_ind)
            actual_train_pool.push_back(train_pool.get(permutation[obj_ind]));
        model->fit(actual_train_pool);
        if (log_file.size() > 0) {
            std::vector<int> predictions = model->predict(test_pool);
            stream << metric(test_pool, predictions) << std::endl;
        }
    }

    if (log_file.size() > 0) {
        stream << std::endl;
        stream.close();
    }
    return model;
}
