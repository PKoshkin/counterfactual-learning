#!/bin/bash

mkdir evaluation/argmax
mkdir evaluation/weighted
python2 ../evaluation/run_evaluation.py --predictions_folder res --data_folder ../../../sorted_data/small_days_data --out_folder evaluation/argmax --type argmax_regression
python2 ../evaluation/run_evaluation.py --predictions_folder res --data_folder ../../../sorted_data/small_days_data --out_folder evaluation/weighted --type weighted_regression
