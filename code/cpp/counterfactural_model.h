#pragma once

#include "utils.h"
#include "model.h"


class CounterfacturalModel {
private:
    std::vector<BaseModel*> models;
public:
    CounterfacturalModel(std::vector<BaseModel*> models);
    void fit(const Pool& train_pool);
    std::vector<int> predict(const Pool& test_pool) const;
    std::vector<std::vector<double>> predict_proba(const Pool& test_pool) const;
    std::vector<double> predict_proba(const Object& object) const;
};
