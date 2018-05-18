#/bin/bash

cd ..
python2 run.py --data_folder ../../../sorted_data/small_days_data --out_folder tests/stacking_res --model catboost --type linear_stacking --verbose --fast --linear_predictions tests/res/features_from_0_to_100 tests/res/features_from_200_to_300 tests/res/features_from_700_to_800 --max_clicks 5
