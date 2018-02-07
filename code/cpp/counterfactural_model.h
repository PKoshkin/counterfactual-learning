#pragma once

#include "utils.h"
#include "model.h"


class CounterfacturalModel {
public:
    virtual void fit(const Pool& train_pool) = 0;
    virtual std::vector<int> predict(const Pool& test_pool) const = 0;
    virtual std::vector<std::vector<double>> predict_proba(const Pool& test_pool) const = 0;
    virtual std::vector<double> predict_proba(const Object& object) const = 0;
};


class ElevenRegressions : public CounterfacturalModel{
private:
    std::vector<BaseModel*> models;
public:
    ElevenRegressions(std::vector<BaseModel*> models);
    virtual void fit(const Pool& train_pool);
    virtual std::vector<int> predict(const Pool& test_pool) const;
    virtual std::vector<std::vector<double>> predict_proba(const Pool& test_pool) const;
    virtual std::vector<double> predict_proba(const Object& object) const;
};
