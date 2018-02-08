#include "xgboost_model.h"

#include <algorithm>


XGBoostModel::XGBoostModel(uint16_t iteration_number)
    : iteration_number(iteration_number) {}


void XGBoostModel::fit(const Matrix& features, const std::vector<double>& scores) {
    std::cout << "\nStart preparing data" << std::endl;

    int rows = features.size();
    int cols = features[0].size();
    float* train_features = new float[rows * cols];
    float* train_labels = new float[rows];
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col)
            train_features[row * cols + col] = static_cast<float>(features[row][col]);
        train_labels[row] = static_cast<float>(scores[row]);
    }

    DMatrixHandle train_data;
    XGDMatrixCreateFromMat(train_features, rows, cols, -1, &train_data);
    XGDMatrixSetFloatInfo(train_data, "label", train_labels, rows);

    XGBoosterCreate(&train_data, 1, &booster);
    XGBoosterSetParam(booster, "booster", "gbtree");
    XGBoosterSetParam(booster, "objective", "reg:linear");
    XGBoosterSetParam(booster, "eta", "0.3");
    XGBoosterSetParam(booster, "gamma", "0");
    XGBoosterSetParam(booster, "max_depth", "6");
    XGBoosterSetParam(booster, "min_child_weight", "1");
    XGBoosterSetParam(booster, "max_delta_step", "0");
    XGBoosterSetParam(booster, "subsample", "1");
    XGBoosterSetParam(booster, "colsample_bytree", "1");
    XGBoosterSetParam(booster, "colsample_bylevel", "1");
    XGBoosterSetParam(booster, "lambda", "1");
    XGBoosterSetParam(booster, "alpha", "0");
    XGBoosterSetParam(booster, "scale_pos_weight", "1");
    XGBoosterSetParam(booster, "refresh_leaf", "1");
    XGBoosterSetParam(booster, "base_score", "0");
    XGBoosterSetParam(booster, "eval_metric", "rmse");
    XGBoosterSetParam(booster, "silent", "1");
    XGDMatrixFree(train_data);

    std::cout << "\nStart training" << std::endl;

    for (int iteration = 0; iteration < iteration_number; ++iteration)
        XGBoosterUpdateOneIter(booster, iteration, train_data);

    delete[] train_features;
    delete[] train_labels;
    XGDMatrixFree(train_data);

    std::cout << "\nEnd training" << std::endl;
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
