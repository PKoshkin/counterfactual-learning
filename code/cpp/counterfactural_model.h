#pragma once

#include "utils.h"
#include "model.h"


class CounterfacturalModel {
public:
    virtual void fit(const Pool& train_pool) = 0;
    virtual std::vector<int> predict(const Pool& test_pool) const = 0;
    virtual std::vector<double> predict_proba(const Object& object) const = 0;
    std::vector<std::vector<double>> predict_proba(const Pool& test_pool) const;
};


class ElevenRegressionsModel : public CounterfacturalModel {
private:
    std::vector<BaseModel*> models;
public:
    ElevenRegressionsModel(std::vector<BaseModel*> models);
    virtual void fit(const Pool& train_pool);
    virtual std::vector<int> predict(const Pool& test_pool) const;
    virtual std::vector<double> predict_proba(const Object& object) const;
};


class PositionToFeaturesModel : public CounterfacturalModel {
private:
    BaseModel* model;
    std::vector<int> positions;
public:
    PositionToFeaturesModel(BaseModel* model, const std::vector<int>& positions);
    virtual void fit(const Pool& train_pool);
    virtual std::vector<int> predict(const Pool& test_pool) const;
    virtual std::vector<double> predict_proba(const Object& object) const;
};
