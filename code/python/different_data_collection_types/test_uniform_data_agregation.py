import sys
sys.path.append('../general')
from metric import metric
sys.path.append('../simple_regression')
from test_regression import get_positions, make_features


def get_deault_and_uniform_metric(pool, default_model, uniorm_model):
    train_pool, test_pool = pool.train_test_split()
    uniform_train_pool = train_pool.get_uniform_pool(len(train_pool.features))

    default_model.fit(
        make_features(train_pool.features, train_pool.positions), train_pool.targets
    )
    uniorm_model.fit(
        make_features(uniform_train_pool.features, uniform_train_pool.positions), uniform_train_pool.targets
    )
    default_positions = get_positions(test_pool.features, default_model)
    uniform_positions = get_positions(test_pool.features, uniorm_model)

    return (
        metric(default_positions, test_pool.positions, test_pool.targets, test_pool.probas),
        metric(uniform_positions, test_pool.positions, test_pool.targets, test_pool.probas)
    )


def test_uniform_data_agregation(default_models, uniorm_models, pool):
    default_scores, uniorm_scores = [], []
    for default_model, uniorm_model in zip(default_models, uniorm_models):
        default_score, uniorm_score = get_deault_and_uniform_metric(pool, default_model, uniorm_model)
        default_scores.append(default_score)
        uniorm_scores.append(uniorm_score)
    return default_scores, uniorm_scores
