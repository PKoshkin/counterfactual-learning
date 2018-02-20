#include "xgboost_model.h"

#include <algorithm>
#include <new>
#include <exception>


void XGBoosterRAIIHandle::reset() {if (booster) XGBoosterFree(booster);}
BoosterHandle* XGBoosterRAIIHandle::get_ptr() {return &booster;}
BoosterHandle XGBoosterRAIIHandle::get() const {return booster;}
XGBoosterRAIIHandle::XGBoosterRAIIHandle(BoosterHandle booster) : booster(booster) {}
XGBoosterRAIIHandle::~XGBoosterRAIIHandle() {if (booster) XGBoosterFree(booster);}


class DMatrixRAIIHandle {
private:
    DMatrixHandle data;
public:
    DMatrixHandle* get() {return &data;}
    DMatrixRAIIHandle(DMatrixHandle data = nullptr) : data(data) {}
    ~DMatrixRAIIHandle() {if (data) XGDMatrixFree(data);}
};


XGBoostModel::XGBoostModel(uint16_t iteration_number, const BoosterParams& booster_params)
    : iteration_number(iteration_number), booster_params(booster_params), booster() {}


void XGBoostModel::fit(const Matrix& features, const std::vector<double>& scores) {
    std::shared_ptr<float> train_features;
    std::shared_ptr<float> train_labels;

    while (true) {
        try {
            booster.reset();
            DMatrixRAIIHandle train_data;
            int rows = features.size();
            int cols = features[0].size();

            train_features = std::shared_ptr<float>(new float[rows * cols]);
            train_labels = std::shared_ptr<float>(new float[rows]);

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    int index = row * cols + col;
                    train_features.get()[index] = static_cast<float>(features[row][col]);
                }
                train_labels.get()[row] = static_cast<float>(scores[row]);
            }

            XGDMatrixCreateFromMat(train_features.get(), rows, cols, -1, train_data.get());
            XGDMatrixSetFloatInfo(train_data.get(), "label", train_labels.get(), rows);

            XGBoosterCreate(train_data.get(), 1, booster.get_ptr());
            XGBoosterSetParam(booster.get(), "eta", "0.1");
            XGBoosterSetParam(booster.get(), "max_depth", "3");
            XGBoosterSetParam(booster.get(), "base_score", "0");
            XGBoosterSetParam(booster.get(), "silent", "1");

            for (const auto& param: booster_params)
                XGBoosterSetParam(booster.get(), param.first.c_str(), param.second.c_str());

            for (int iteration = 0; iteration < iteration_number; ++iteration)
                XGBoosterUpdateOneIter(booster.get(), iteration, *train_data.get());

            return;
        }
        catch (std::bad_alloc err) {}
    }
}


double XGBoostModel::predict(const std::vector<double>& features) const {
    int rows = 1;
    int cols = features.size();
    float test_features[rows][cols];

    for (int col = 0; col < cols; ++col)
        test_features[0][col] = static_cast<float>(features[col]);

    DMatrixHandle test_data;
    XGDMatrixCreateFromMat((float *) test_features, rows, cols, -1, &test_data);

    bst_ulong out_len;
    const float * prediction;
    XGBoosterPredict(booster.get(), test_data, 0, 0, &out_len, &prediction);

    XGDMatrixFree(test_data);
    return prediction[0];
}


std::vector<double> XGBoostModel::predict_proba(const std::vector<double>& features) const {
    int rows = 1;
    int cols = features.size();
    float test_features[rows][cols];

    for (int col = 0; col < cols; ++col)
        test_features[0][col] = static_cast<float>(features[col]);

    DMatrixHandle test_data;
    XGDMatrixCreateFromMat((float *) test_features, rows, cols, -1, &test_data);

    bst_ulong out_len;
    const float * prediction;
    XGBoosterPredict(booster.get(), test_data, 0, 0, &out_len, &prediction);

    XGDMatrixFree(test_data);

    std::vector<double> result(out_len);
    for (int i = 0; i < out_len; ++i)
        result[i] = prediction[i];
    return result;
}
