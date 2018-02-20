import numpy as np


class CounterfactualModelError(Exception):
    pass


class ClassificationCounterfactualModel:
    NONE_POSITION = 10

    def __init__(self, models):
        self.models = models

    def fit(self, train_pools, kwargs_list):
        if len(train_pools) != len(self.models):
            raise CounterfactualModelError('Wrong pools number')
        for pool, model, kwargs in zip(train_pools, self.models, kwargs_list):
            model.fit(pool.features, pool.classification_labels, **kwargs)

    def predict(self, test_pool):
        predictions = np.array([
            model.predict(test_pool.features)
            for model in self.models
        ]).T

        mask = np.any(predictions >= 2, axis=1)
        show_position = np.argmax(
            predictions, axis=1
        )
        return show_position * mask + ClassificationCounterfactualModel.NONE_POSITION * (1 - mask)
