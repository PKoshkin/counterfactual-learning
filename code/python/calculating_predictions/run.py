import argparse
from calculate_predictions import calculate_predictions
from linear_stacking import calculate_classification_stacked_on_linear_predictions
from catboost import CatBoostRegressor, CatBoostClassifier
from linear import calculate_linear


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--out_folder", required=True)
    parser.add_argument("--model", required=True, help="catboost")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--first_feature", type=int, default=0)
    parser.add_argument("--last_feature", type=int, default=-1)
    parser.add_argument("--train_day", type=int, required=True)
    parser.set_defaults(additional_features=None)

    type_adder = parser.add_subparsers(dest="type")

    regression_parser = type_adder.add_parser("regression")
    regression_parser.set_defaults(func=calculate_predictions)
    regression_parser.set_defaults(model_constructor=lambda verbose: CatBoostRegressor(verbose=verbose))

    classification_parser = type_adder.add_parser("classification")
    classification_parser.add_argument("--max_clicks", type=int, required=True)
    classification_parser.set_defaults(func=calculate_predictions)
    classification_parser.set_defaults(
        model_constructor=lambda verbose, max_clicks: CatBoostClassifier(
            verbose=verbose,
            loss_function='MultiClass',
            classes_count=max_clicks + 2
        )
    )

    binary_classification_parser = type_adder.add_parser("binary_classification")
    binary_classification_parser.add_argument("--threshold", type=float, required=True)
    binary_classification_parser.set_defaults(func=calculate_predictions)
    binary_classification_parser.set_defaults(model_constructor=lambda verbose: CatBoostClassifier(verbose=verbose))

    linear_parser = type_adder.add_parser("linear")
    linear_parser.add_argument("--step", type=int)
    linear_parser.set_defaults(func=calculate_linear)

    linear_stacking_parser = type_adder.add_parser("linear_stacking")
    linear_stacking_parser.add_argument(
        "--linear_predictions",
        type=str,
        nargs='*',
        help="list of folders with linear regression predictions"
    )
    linear_stacking_parser.add_argument("--max_clicks", type=int, required=True)
    linear_stacking_parser.set_defaults(func=calculate_classification_stacked_on_linear_predictions)
    linear_stacking_parser.set_defaults(
        model_constructor=lambda verbose, max_clicks: CatBoostClassifier(
            verbose=verbose,
            loss_function='MultiClass',
            classes_count=max_clicks + 2
        )
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    run()
