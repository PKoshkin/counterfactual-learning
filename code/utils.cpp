#include "utils.h"

#include <utility>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "json.hpp"

using json = nlohmann::json;

int Pool::size() const {
    return metrics.size();
}

void Pool::reserve(int size) {
    factors.reserve(size);
    positions.reserve(size);
    metrics.reserve(size);
    probas.reserve(size);
}

std::pair<Pool, Pool> Pool::train_test_split(double train_share) {
    int waterline = int(factors.size() * train_share);
    Pool train;
    Pool test;

    train.is_train = 1;
    train.factors.assign(factors.begin(), factors.begin() + waterline);
    for (int index = 0; index < waterline; ++index) {
	train.factors[index].resize(FACTORS_LEN + POSITIONS.size(), 0);

	if (positions[index] == 100)
	    train.factors[index][FACTORS_LEN + 10] = 1;
	else
	    train.factors[index][FACTORS_LEN + positions[index]] = 1;

    }

    train.metrics.assign(metrics.begin(), metrics.begin() + waterline);
    train.probas.assign(probas.begin(), probas.begin() + waterline);

    test.is_train = 0;
    test.factors.reserve((factors.size() - waterline) * POSITIONS.size());

    for (int index = waterline; index < factors.size(); ++index) {
	for (int pos_ind = 0; pos_ind < POSITIONS.size(); ++pos_ind) {
	    test.factors.push_back(factors[index]);
	    test.factors[(index - waterline) * POSITIONS.size() + pos_ind].resize(FACTORS_LEN + POSITIONS.size(), 0);
	    test.factors[(index - waterline) * POSITIONS.size() + pos_ind][FACTORS_LEN + pos_ind] = 1;
	}
    }

    test.metrics.assign(metrics.begin() + waterline, metrics.end());
    test.probas.assign(probas.begin() + waterline, probas.end());
    test.positions.assign(positions.begin() + waterline, positions.end());
    return std::pair<Pool, Pool>(train, test);
}

std::pair<int, int> get_position(const std::string& line, int search_start_position = 0) {
    int pos_position = line.find("rnd_pos", search_start_position) + strlen("rnd_pos") + 3;
    if (line[pos_position + 1] == '0') {
	return {100, pos_position + 4};
    } else {
	return {line[pos_position] - '0', pos_position + 2};
    }
}

std::pair<double, int> get_probability(const std::string& line, int search_start_position = 0) {
    int start_position = line.find("p", search_start_position) + strlen("p") + 3;
    int end_position = line.find(",", start_position);
    return {atof(line.substr(start_position, end_position).c_str()), end_position + 2};
}

std::pair<std::vector<double>, int> get_vector(const std::string& line, std::string attribute,
					       int search_start_position = 0) {
    std::vector<double> result;
    int start_position = line.find(attribute) + attribute.size() + 4;
    if (line.find("null", start_position) != std::string::npos)
	return {{}, start_position + 4};

    int end_position = line.find("]", start_position);

    while (start_position < end_position) {
	int comma_position = line.find(",", start_position);
	result.push_back(atof(line.substr(start_position, comma_position).c_str()));
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
