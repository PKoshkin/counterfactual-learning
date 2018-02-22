#include "pool.h"

#include <stdexcept>


int Pool::size() const {
    return metrics.size();
}


void Pool::clear() {
    factors.clear();
    positions.clear();
    metrics.clear();
    probas.clear();
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


void Pool::assign(
    const Pool& pool,
    std::vector<int>::const_iterator begin,
    std::vector<int>::const_iterator end
) {
    resize(end - begin);

    for (auto it = begin; it != end; ++it)
        set(it - begin, pool.get(*it));
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
    probas.push_back(obj.proba);
    factors.push_back(obj.factors);
}


void Pool::erase(uint32_t index) {
    positions.erase(positions.begin() + index);
    probas.erase(probas.begin() + index);
    metrics.erase(metrics.begin() + index);
    factors.erase(factors.begin() + index);
}


void Pool::set(uint32_t index, const Object& obj) {
    if (index < size()) {
        positions[index] = obj.position;
        metrics[index] = obj.metric;
        probas[index] = obj.proba;
        factors[index] = obj.factors;
    } else {
        throw std::runtime_error("Pool index out of range");
    }
}


void Pool::set(
    uint32_t index,
    int position,
    double metric,
    double proba,
    const std::vector<double>& factors
) {
    if (index < size()) {
        positions[index] = position;
        metrics[index] = metric;
        probas[index] = proba;
        this->factors[index] = factors;
    } else {
        throw std::runtime_error("Pool index out of range");
    }
}


Object Pool::get(uint32_t index) const {
    if (index < size())
        return Object(positions[index], metrics[index], probas[index], factors[index]);
    else
        throw std::runtime_error("Pool index out of range");
}
