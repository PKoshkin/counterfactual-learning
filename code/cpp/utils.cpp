#include "utils.h"

#include <stdexcept>
#include <fstream>
#include <string>
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


std::pair<Pool, Pool> train_test_split(const Pool& pool, double train_share, uint32_t random_seed) {
    int waterline = int(pool.size() * train_share);
    Pool train;
    Pool test;

    if (random_seed == -1) {
        train.assign(pool, waterline);
        test.assign(pool, waterline, pool.size());
    } else {
        std::srand(random_seed);
        std::vector<int> permutation = get_permutation(pool.size());

        train.assign(pool, permutation.begin(), permutation.begin() + waterline);
        test.assign(pool, permutation.begin() + waterline, permutation.end());
    }

    return {train, test};
}


double get_krkopt_feature(const std::string& line, size_t index_start, size_t index_end) {
    if ('a' <= line[index_start] && line[index_start] <= 'h')
        return line[index_start] - 'a';
    else
        return line[index_start] - '0';
}


double get_krkopt_isfourteen_answer(const std::string& line, size_t index) {
    if (line.substr(index) == KRKOPT_ANSWERS[15])
        return 0;
    else
        return 1;
}


double get_krkopt_raw_answer(const std::string& line, size_t index) {
    for (int answer_index = 0; answer_index < KRKOPT_ANSWERS.size(); ++answer_index)
        if (KRKOPT_ANSWERS[answer_index] == line.substr(index))
            return answer_index;
}


double get_float_feature(const std::string& line, size_t index_start, size_t index_end) {
    if ('0' > line[index_start] || line[index_start] > '9')
        return -1;
    else
        return std::stof(line.substr(index_start));
}


double get_breast_answer(const std::string& line, size_t index) {
    if (line[index] == 'b')
        return 0;
    else
        return 1;
}


double get_ionosphere_answer(const std::string& line, size_t index) {
    if (line[index] == 'g')
        return 0;
    else
        return 1;
}


double get_diabetes_answer(const std::string& line, size_t index) {
    if (line[index + 7] == 'n')
        return 0;
    else
        return 1;
}

double get_sonar_answer(const std::string& line, size_t index) {
    if (line[index] == 'R')
        return 0;
    else
        return 1;
}


Pool get_test_classification_pool(
    std::string file_name,
    GetAnswer* get_answer,
    GetFeature* get_feature,
    int start_size
) {
    std::ifstream infile(file_name);
    std::string line;
    Pool result;
    result.reserve(start_size);

    size_t line_ind = 0;
    while (std::getline(infile, line)) {
        std::vector<double> factors;
        size_t index = 0;
        while (index < line.size()) {
            size_t comma_ind = line.find(',', index);
            if (comma_ind == std::string::npos)
                break;
            factors.push_back(get_feature(line, index, comma_ind));
            index = comma_ind + 1;
        }
        double answer = get_answer(line, index);
        result.push_back(Object(0, answer, 1, factors));
        if (line_ind < 10) {
            for (auto it: factors)
                std::cout << it << " ";
            std::cout << answer << std::endl;
        }
        line_ind++;
    }

    return result;
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

std::pair<std::vector<double>, int> get_vector(
        const std::string& line,
        std::string attribute,
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


std::vector<std::vector<int>> get_permutations(
        uint32_t permutations_number,
        uint32_t permutation_size,
        uint32_t random_seed) {
    std::srand(random_seed);
    std::vector<std::vector<int>> permutations(permutations_number);
    for (auto& permutation: permutations)
        permutation = get_permutation(permutation_size);
    return permutations;
}


void softmax(std::vector<double>& array) {
    double exp_sum = 0;
    double minimum = *std::min_element(array.begin(), array.end());

    for (auto& iter: array) {
        iter = std::exp(iter - minimum);
        exp_sum += iter;
    }

    for (auto& iter: array)
        iter /= exp_sum;
}


uint32_t get_uint_param(
    int argc,
    char* argv[],
    uint16_t index,
    uint32_t default_value,
    const char name[]
) {
    uint32_t param;
    if (argc > index)
        param = atoi(argv[index]);
    else
        param = default_value;
    std::cout << name << ": " << param << std::endl;
    return param;
}
