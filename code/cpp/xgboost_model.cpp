#include "xgboost_model.h"

#include <algorithm>


XGBoostModel::XGBoostModel(uint16_t iteration_number, uint16_t batch_size)
    : iteration_number(iteration_number), batch_size(batch_size) {}


void XGBoostModel::fit(const Matrix& features, const std::vector<double>& scores) {
    std::cout << "\nStart preparing data" << std::endl;
    DMatrixHandle train_data;
    const int cols = features[0].size();
    float train_features[batch_size][cols] = {0};
    float train_labels[batch_size] = {0};

    XGDMatrixCreateFromMat((float *) train_features, batch_size, cols, -1, &train_data);
    XGDMatrixSetFloatInfo(train_data, "label", train_labels, batch_size);

    XGBoosterCreate(&train_data, 1, &booster);
    XGBoosterSetParam(booster, "objective", "reg:linear");
    XGBoosterSetParam(booster, "eta", "0.1");
    XGBoosterSetParam(booster, "silent", "1");

    std::cout << "\nStart training" << std::endl;

    for (int iteration = 0; iteration < iteration_number; ++iteration) {
        for (int batch_start_ind = 0; batch_start_ind < features.size(); batch_start_ind += batch_size) {
            int rows = std::min(batch_size, static_cast<uint16_t>(features.size() - batch_start_ind));
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col)
                    train_features[row][col] = static_cast<float>(features[batch_start_ind + row][col]);
                train_labels[row] = static_cast<float>(scores[batch_start_ind + row]);
            }

            XGDMatrixCreateFromMat((float *) train_features, rows, cols, -1, &train_data);
            XGDMatrixSetFloatInfo(train_data, "label", train_labels, rows);

            XGBoosterUpdateOneIter(booster, iteration, train_data);
        }
    }

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
