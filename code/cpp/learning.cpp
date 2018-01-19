#include "learning.h"
#include <iostream>
#include <cstring>

BoosterHandle train(const Pool& train_pool, int iteration_number, int batch_size) {
    std::cout << "\nPrepare data for training\n" << std::endl;

    DMatrixHandle train_data;
    int rows = train_pool.size();
    int cols = FACTORS_LEN + 1;
    float train_features[rows][cols];
    float train_labels[rows];

    for (int row = 0; row < rows; ++row) {
	for (int col = 0; col < cols; ++col)
	    train_features[row][col] = train_pool.factors[row][col];
	train_labels[row] = train_pool.metrics[row];
    }

    XGDMatrixCreateFromMat((float *) train_features, rows, cols, -1, &train_data);
    XGDMatrixSetFloatInfo(train_data, "label", train_labels, rows);

    BoosterHandle booster;
    XGBoosterCreate(&train_data, 1, &booster);
    XGBoosterSetParam(booster, "objective", "reg:linear");
    XGBoosterSetParam(booster, "eta", "0.1");
    XGBoosterSetParam(booster, "silent", "1");

    std::cout << "\nStart training\n" << std::endl;

    for (int iteration = 0; iteration < iteration_number; ++iteration)
	XGBoosterUpdateOneIter(booster, iteration, train_data);

    XGDMatrixFree(train_data);

    std::cout << "\nEnd training\n" << std::endl;
    return booster;
}

float test(const Pool& test_pool, const BoosterHandle& booster, int batch_size) {
    std::cout << "\nStart testing\n" << std::endl;
    float result_metric = 0;

    for (int batch_start_ind = 0; batch_start_ind < test_pool.size(); batch_start_ind += batch_size) {
	int pos_num = POSITIONS.size();
	int rows = batch_size * pos_num;
	int cols = FACTORS_LEN + 1;
	float test_features[rows][cols];

	for (int row = 0; row < rows; ++row)
	    for (int col = 0; col < cols; ++col)
		test_features[row][col] = test_pool.factors[batch_start_ind + row][col];
	DMatrixHandle test_data;
	XGDMatrixCreateFromMat((float *) test_features, rows, cols, -1, &test_data);

	bst_ulong out_len;
	const float * prediction;
	XGBoosterPredict(booster, test_data, 0, 0, &out_len, &prediction);

	for (int test_ind = 0; test_ind < batch_size; ++test_ind) {
	    const float * curr_prediction = prediction + test_ind * pos_num;
	    int real_pos = test_pool.positions[batch_start_ind + test_ind];
	    bool guessed = true;

	    for (int position_ind = 0; position_ind < pos_num; ++position_ind)
		if (curr_prediction[position_ind] > curr_prediction[real_pos])
		    guessed = false;

	    if (guessed) {
		float proba = test_pool.probas[batch_start_ind + test_ind];
		float metric = test_pool.metrics[batch_start_ind + test_ind];
		result_metric += metric / proba;
	    }
	}

	XGDMatrixFree(test_data);
    }

    std::cout << "\nEnd testing\n" << std::endl;
    return result_metric / test_pool.size();
}
