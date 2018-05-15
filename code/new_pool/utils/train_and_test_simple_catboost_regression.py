from catboost import CatBoostRegressor
from run_train_and_test_simple_regression import run


if __name__ == "__main__":
    run(lambda verbose: CatBoostRegressor(verbose=verbose))
