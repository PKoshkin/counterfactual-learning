#/bin/bash

python2 ../calculating_predictions/run.py --data_folder ../../../sorted_data/small_days_data --out_folder res --model catboost --type binary_classification --verbose --fast --threshold 1
