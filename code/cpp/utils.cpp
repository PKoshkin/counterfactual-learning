#include "utils.h"

#include <cmath>
#include <algorithm>
#include <utility>
#include <cassert>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cstdlib>


Object::Object(int position, double metric, double proba, const std::vector<double> factors)
    : position(position), metric(metric), proba(proba), factors(factors) {}


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
    std::cout << "\nStart reading pool" << std::endl;
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
    std::cout << "\nEnd reading pool" << std::endl;
    return result;
}


std::vector<int> get_permutation(int size) {
    std::vector<int> result(size);
    for (int i = 0; i < size; ++i)
    result[i] = i;
    std::random_shuffle(result.begin(), result.end());
    return result;
}


void softmax(std::vector<double>& array) {
    double exp_sum = 0;
    for (auto& iter: array) {
        iter = std::exp(iter);
        exp_sum += iter;
    }

    for (auto& iter: array)
        iter /= exp_sum;
}
