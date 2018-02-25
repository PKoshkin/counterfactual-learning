from sklearn.decomposition import PCA
import sys
sys.path.append('../general')
from metric import metric
sys.path.append('../simple_regression')
from test_regression import get_positions, make_features


def get_metric_with_pca(pool, model, dimention):
    train_pool, test_pool = pool.train_test_split()
    pca = PCA(500)
    pca.fit(train_pool.features)
    train_pool.set_features(pca.transform(train_pool.features))
    test_pool.set_features(pca.transform(test_pool.features))
    model.fit(make_features(train_pool.features, train_pool.positions), train_pool.targets)
    positions = get_positions(test_pool.features, model)

    return metric(positions, test_pool.positions, test_pool.targets, test_pool.probas)


def test_regression_with_pca(models, pool, dimentions):
    scores = [
        get_metric_with_pca(pool, model, dimention)
        for model, dimention in zip(models, dimentions)
    ]
    return scores
