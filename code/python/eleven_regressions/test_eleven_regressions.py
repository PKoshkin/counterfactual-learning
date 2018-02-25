import numpy as np
from counterfactual_model import CounterfactualModel
import sys
sys.path.append('../general')
from metric import metric
from constants import POSITION_VARIANTS


def test_eleven_regressions(model_constructior, pool, runs=100):
    scores = []
    for i in range(runs):
        train_pool, test_pool = pool.train_test_split()
        train_pools = train_pool.split_by_position()
        model = CounterfactualModel([model_constructior() for _ in POSITION_VARIANTS])
        model.fit(train_pools)
        prediction = model.predict(test_pool)
        scores.append(metric(prediction, test_pool.positions, test_pool.targets, test_pool.probas))
    return np.array(scores)
