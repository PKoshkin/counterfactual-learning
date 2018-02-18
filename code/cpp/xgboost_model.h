#pragma once

#include "model.h"

#include <xgboost/c_api.h>


typedef std::vector<std::pair<std::string, std::string>> BoosterParams;


class XGBoostModel : public BaseModel {
public:
    virtual double predict(const std::vector<double>& features) const;
    std::vector<double> predict_proba(const std::vector<double>& features) const;
    virtual void fit(const Matrix& features, const std::vector<double>& scores);
    XGBoostModel(uint16_t iteration_number, const BoosterParams& booster_params = {});
    ~XGBoostModel();
private:
    BoosterParams booster_params;
    BoosterHandle booster;
    uint16_t iteration_number;
};
