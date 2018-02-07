#pragma once

#include "model.h"

#include <xgboost/c_api.h>


class XGBoostModel : public BaseModel {
public:
    virtual double predict(const std::vector<double>& features) const;
    virtual void fit(const Matrix& features, const std::vector<double>& scores);
    XGBoostModel(uint16_t iteration_number);
private:
    BoosterHandle booster;
    uint16_t iteration_number;
};
