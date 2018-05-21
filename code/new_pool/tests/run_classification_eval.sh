#!/bin/bash

mkdir evaluation/argmax
mkdir evaluation/exp_weighted
mkdir evaluation/argmax_weighted
mkdir evaluation/exp_argmax
python2 ../evaluation/run_evaluation.py --predictions_folder res --data_folder ../../../sorted_data/small_days_data --out_folder evaluation/argmax --type argmax_classification
python2 ../evaluation/run_evaluation.py --predictions_folder res --data_folder ../../../sorted_data/small_days_data --out_folder evaluation/exp_weighted --type exprect_weighted_classification
python2 ../evaluation/run_evaluation.py --predictions_folder res --data_folder ../../../sorted_data/small_days_data --out_folder evaluation/argmax_weighted --type argmax_weighted_classification
python2 ../evaluation/run_evaluation.py --predictions_folder res --data_folder ../../../sorted_data/small_days_data --out_folder evaluation/exp_argmax --type exprect_argmax_classification
