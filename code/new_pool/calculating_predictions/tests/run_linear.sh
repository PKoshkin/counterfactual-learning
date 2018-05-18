#/bin/bash

cd ..
python2 run.py --data_folder ../../../sorted_data/small_days_data --out_folder tests/res --model svr --type linear --verbose --fast --step 100
