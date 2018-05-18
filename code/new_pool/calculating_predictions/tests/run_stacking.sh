#/bin/bash

cd ..
python2 run.py --data_folder ../../../sorted_data/small_days_data --out_folder tests/stacking_res --model catboost --type linear_stacking --verbose --fast --max_clicks 5 --linear_predictions tests/res/features_from_0_to_100 tests/res/features_from_100_to_200 tests/res/features_from_200_to_300 tests/res/features_from_300_to_400 tests/res/features_from_400_to_500 tests/res/features_from_500_to_600 tests/res/features_from_600_to_700 tests/res/features_from_700_to_800 tests/res/features_from_800_to_900 tests/res/features_from_900_to_1000 tests/res/features_from_1000_to_1100 
