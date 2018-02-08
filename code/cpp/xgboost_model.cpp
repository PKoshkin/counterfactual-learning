#include "xgboost_model.h"


XGBoostModel::XGBoostModel(uint16_t iteration_number) : iteration_number(iteration_number) {}


void XGBoostModel::fit(const Matrix& features, const std::vector<double>& scores) {
    DMatrixHandle train_data;
    const int rows = features.size();
    const int cols = features[0].size();
    float train_features[rows][cols];
    float train_labels[rows];

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col)
            train_features[row][col] = static_cast<float>(features[row][col]);
        train_labels[row] = static_cast<float>(scores[row]);
    }

    XGDMatrixCreateFromMat((float *) train_features, rows, cols, -1, &train_data);
    XGDMatrixSetFloatInfo(train_data, "label", train_labels, rows);

    XGBoosterCreate(&train_data, 1, &booster);
    XGBoosterSetParam(booster, "objective", "reg:linear");
    XGBoosterSetParam(booster, "eta", "0.1");
    XGBoosterSetParam(booster, "silent", "1");

    for (int iteration = 0; iteration < iteration_number; ++iteration)
	XGBoosterUpdateOneIter(booster, iteration, train_data);

    XGDMatrixFree(train_data);
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
