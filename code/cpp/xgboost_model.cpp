#include "xgboost_model.h"

#include <algorithm>
#include <new>
#include <exception>


XGBoostModel::XGBoostModel(uint16_t iteration_number, const BoosterParams& booster_params)
    : iteration_number(iteration_number), booster_params(booster_params), booster(nullptr) {}


void XGBoostModel::fit(const Matrix& features, const std::vector<double>& scores) {
    bool success = false;
    float* train_features;
    float* train_labels;
    DMatrixHandle train_data;

    while (!success) {
        try {
            int rows = features.size();
            int cols = features[0].size();

            train_features = new float[rows * cols];
            train_labels = new float[rows];

            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col)
                    train_features[row * cols + col] = static_cast<float>(features[row][col]);
                train_labels[row] = static_cast<float>(scores[row]);
            }

            XGDMatrixCreateFromMat((float *) train_features, rows, cols, -1, &train_data);
            XGDMatrixSetFloatInfo(train_data, "label", train_labels, rows);

            if (!booster)
                XGBoosterFree(booster);
            XGBoosterCreate(&train_data, 1, &booster);
            XGBoosterSetParam(booster, "eta", "0.1");
            XGBoosterSetParam(booster, "max_depth", "3");
            XGBoosterSetParam(booster, "base_score", "0");
            XGBoosterSetParam(booster, "silent", "1");

            for (auto param: booster_params)
                XGBoosterSetParam(booster, param.first.c_str(), param.second.c_str());

            for (int iteration = 0; iteration < iteration_number; ++iteration)
                XGBoosterUpdateOneIter(booster, iteration, train_data);

            delete[] train_features;
            delete[] train_labels;
            XGDMatrixFree(train_data);
        }
        catch (std::bad_alloc err) {
            try {
                delete[] train_features;
                delete[] train_labels;
                XGDMatrixFree(train_data);
                if (!booster) {
                    XGBoosterFree(booster);
                    booster = nullptr;
                }
            }
            catch (std::exception _err) {}
        }

        success = true;
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
    XGBoosterPredict(booster, test_data, 0, 0, &out_len, &prediction);

    XGDMatrixFree(test_data);
    return prediction[0];
}


XGBoostModel::~XGBoostModel() {
    XGBoosterFree(booster);
}
