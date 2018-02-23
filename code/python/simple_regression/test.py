import sys
sys.path.append('../general')
from utils import predict_positions
from metric import metric


def test_regression(model, pool, trys=100):
    scores = []
    for i in range(trys):
        train_pool, test_pool = pool.train_test_split()
        model.fit(train_pool.features_with_positions, train_pool.targets)
        positions = predict_positions(test_pool.features, model)
        scores.append(metric(positions, test_pool.positions, test_pool.targets, test_pool.probas))
    return scores
