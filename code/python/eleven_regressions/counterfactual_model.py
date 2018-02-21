import numpy as np


class CounterfactualModelError(Exception):
    pass


class CounterfactualModel:
    NONE_POSITION = 11

    def __init__(self, models):
        self.models = models

    def fit(self, train_pools):
        if len(train_pools) != len(self.models):
            raise CounterfactualModelError('Wrong pools noumber')
        for pool, model in zip(train_pools, self.models):
            model.fit(pool.features, pool.targets)

    def predict(self, test_pool):
        predictions = np.array([
            model.predict(test_pool.features)
            for model in self.models
        ]).T

        return [
            test_pool.POSITIONS[np.argmax(models_predictions)]
            for models_predictions in predictions
        ]
