#pragma once

#include "utils.h"
#include "counterfactural_model.h"
#include "xgboost_model.h"


class SimpleClassification : public CounterfacturalModel {
private:
    XGBoostModel model;
    uint16_t num_class;
public:
    SimpleClassification(uint16_t num_iteration, uint16_t num_class);
    virtual void fit(const Pool& train_pool);
    virtual std::vector<int> predict(const Pool& test_pool) const;
    virtual std::vector<double> predict_proba(const Object& object) const;
};
