#include "utils.h"

#include <utility>
#include <exception>
#include <cassert>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cstdlib>

class index_out_of_range: public runtime_error {
    virtual const char* what() const throw() {
        return "Index out of range";
    }
};


Object::Object(int position, double metric, double proba, const std::vector<double> factors)
    : position(position), metric(metric), proba(proba), factors(factors) {}


int Pool::size() const {
    return metrics.size();
}

void Pool::reserve(int size) {
    factors.reserve(size);
    positions.reserve(size);
    metrics.reserve(size);
    probas.reserve(size);
}


void Pool::resize(int size) {
    factors.resize(size);
    positions.resize(size);
    metrics.resize(size);
    probas.resize(size);
}


void Pool::assign(const Pool& pool, int begin, int end) {
    if (end == -1) {
        end = begin;
        begin = 0;
    }

    factors.assign(pool.factors.begin() + begin, pool.factors.begin() + end);
    positions.assign(pool.positions.begin() + begin, pool.positions.begin() + end);
    probas.assign(pool.probas.begin() + begin, pool.probas.begin() + end);
    metrics.assign(pool.metrics.begin() + begin, pool.metrics.begin() + end);
}


void Pool::assign(const Pool& pool, std::vector<int> indexes, int end = -1) {
    if (end < 0 || end > indexes.size())
        end = indexes.size();

    resize(end);

    for (int ind = 0; ind < end; ++ind)
        set(ind, pool.get(indexes[ind]));
}


std::vector<Pool> Pool::split_by_positions() const {
    std::vector<Pool> result(POSITIONS.size());

    for (int i = 0; i < size(); ++i) {
        int pos_ind = positions[i] == 100 ? 10 : positions[i];

        result[pos_ind].factors.push_back(factors[i]);
        result[pos_ind].metrics.push_back(metrics[i]);
        result[pos_ind].probas.push_back(probas[i]);
        result[pos_ind].positions.push_back(positions[i]);
    }

    return result;
}


void Pool::push_back(const Object& obj) {
    positions.push_back(obj.position);
    metrics.push_back(obj.metric);
    probas.push_back(obj.probas);
    factors.push_back(obj.factors);
}


void Pool::set(int index, const Object& obj) {
    if (index < size()) {
        positions[index] = obj.position;
        metrics[index] = obj.metric;
        probas[index] = obj.proba;
        factors[index] = obj.factors;
    } else {
        throw index_out_of_range();
    }
}


void Pool::set(int index, int position, double metric, double proba, const std::vector<double>& factors) {
    if (index < size()) {
        positions[index] = position;
        metrics[index] = metric;
        probas[index] = proba;
        this->factors[index] = factors;
    } else {
        throw index_out_of_range();
    }
}


Object Pool::get(uint16_t index) {
    if (index < size())
        return Object(positions[index], metrics[index], probas[index], factors[index]);
    else
        throw index_out_of_range();
}


std::pair<Pool, Pool> train_test_split(const Pool& pool, double train_share) {
    int waterline = int(pool.factors.size() * train_share);
    Pool train;
    Pool test;

    train.assign(pool, waterline);

    test.assign(pool, waterline, pool.size());
    return {train, test};
}

std::pair<int, int> get_position(const std::string& line, int search_start_position = 0) {
    int pos_position = line.find("rnd_pos", search_start_position) + strlen("rnd_pos") + 3;
    if (line[pos_position + 1] == '0')
        return {100, pos_position + 3};
    else
        return {line[pos_position] - '0', pos_position + 2};
}

std::pair<double, int> get_probability(const std::string& line, int search_start_position = 0) {
    int start_position = line.find("\"p\"", search_start_position) + strlen("\"p\"") + 3;
    int end_position = line.find(",", start_position);
    double position = atof(line.substr(start_position, end_position - start_position).c_str());
    return {position, end_position + 2};
}

std::pair<std::vector<double>, int> get_vector(const std::string& line, std::string attribute,
                                               int search_start_position = 0) {
    std::vector<double> result;
    int start_position = line.find(attribute) + attribute.size() + 4;
    if (attribute == "images_metric" && line.find("null", start_position) != std::string::npos)
        return {{}, start_position + 4};

    int end_position = line.find("]", start_position);

    while (start_position < end_position) {
        int comma_position = line.find(",", start_position);
        result.push_back(atof(line.substr(start_position, comma_position - start_position).c_str()));
        start_position = comma_position + 2;
    }

    return {result, end_position + 3};
}

Pool get_pool(std::string file_name, int max_line, int start_size) {
    Pool result;
    result.reserve(start_size);

    char json_string[max_line];
    std::string metric_key = "images_metric";
    FILE* file;
    file = fopen(file_name.c_str(), "r");

    while (fgets(json_string, max_line, file)) {
        auto factors_search_reslut = get_vector(std::string(json_string), "factors");
        assert(factors_search_reslut.first.size() >= FACTORS_LEN);
        if (factors_search_reslut.first.size() > FACTORS_LEN)
            factors_search_reslut.first.resize(FACTORS_LEN);
        result.factors.push_back(factors_search_reslut.first);

        auto metrics_search_result = get_vector(std::string(json_string), metric_key, factors_search_reslut.second);
        if (metrics_search_result.first.size() == 0)
            result.metrics.push_back(0);
        else
            result.metrics.push_back(metrics_search_result.first[2] - metrics_search_result.first[1]);

        auto position_search_result = get_position(std::string(json_string), metrics_search_result.second);
        result.positions.push_back(position_search_result.first);

        auto proba_search_result = get_probability(std::string(json_string), position_search_result.second);
        result.probas.push_back(proba_search_result.first);
    }

    fclose(file);
    return result;
}


std::vector<int> get_permutation(int size) {
    std::vector<int> result(size);
    for (int i = 0; i < size; ++i)
    result[i] = i;
    std::random_shuffle(result.begin(), result.end());
    return result;
}
