#include "utils.h"

#include <utility>
#include <iostream>
#include <cstdio>
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

std::pair<Pool, Pool> Pool::train_test_split(float train_share) {
    int waterline = int(factors.size() * train_share);
    Pool train;
    Pool test;

    train.is_train = 1;
    train.factors.assign(factors.begin(), factors.begin() + waterline);
    for (int index = 0; index < waterline; ++index)
	train.factors[index].push_back(positions[index]);

    train.metrics.assign(metrics.begin(), metrics.begin() + waterline);
    train.probas.assign(probas.begin(), probas.begin() + waterline);

    test.is_train = 0;
    test.factors.reserve((factors.size() - waterline) * POSITIONS.size());

    for (int index = waterline; index < factors.size(); ++index)
	for (int pos_ind = 0; pos_ind < POSITIONS.size(); ++pos_ind) {
	    test.factors.push_back(factors[index]);
	    test.factors[(index - waterline) * POSITIONS.size() + pos_ind].push_back(POSITIONS[pos_ind]);
	}

    test.metrics.assign(metrics.begin() + waterline, metrics.end());
    test.probas.assign(probas.begin() + waterline, probas.end());
    test.positions.assign(positions.begin() + waterline, positions.end());
    return std::pair<Pool, Pool>(train, test);
}

Pool get_pool(std::string file_name, int max_line, int start_size) {
    Pool result;
    result.reserve(start_size);

    char json_string[max_line];
    std::string metric_key = "images_metric";
    FILE* file;
    file = fopen(file_name.c_str(), "r");

    while (fgets(json_string, max_line, file)) {
	json object_json = json::parse(json_string);

	std::vector<float> factors = object_json["factors"].get<std::vector<float>>();
	if (factors.size() > FACTORS_LEN)
	    factors.resize(FACTORS_LEN);
	result.factors.push_back(factors);

	if (object_json[metric_key] == nullptr)
	    result.metrics.push_back(0);
	else {
	    float win = object_json[metric_key][2].get<float>();
	    float loss = object_json[metric_key][1].get<float>();
	    result.metrics.push_back(win - loss);
	}

	result.positions.push_back(object_json["rnd_pos"].get<int>());
	result.probas.push_back(object_json["p"].get<float>());
    }

    fclose(file);
    return result;
}
