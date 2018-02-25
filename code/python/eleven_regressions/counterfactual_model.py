import numpy as np
import sys
sys.path.append('../general')
from constants import POSITION_VARIANTS


class CounterfactualModelError(Exception):
    pass


class CounterfactualModel:
    def __init__(self, models):
        self.models = models

    def fit(self, train_pools):
        if len(train_pools) != len(self.models):
            raise CounterfactualModelError('Wrong pools number')
        for pool, model in zip(train_pools, self.models):
            model.fit(pool.features, pool.targets)

    def predict(self, test_pool):
        predictions = np.array([
            model.predict(test_pool.features)
            for model in self.models
        ]).T

        return np.array([
            POSITION_VARIANTS[np.argmax(models_predictions)]
            for models_predictions in predictions
        ])
