import sys
sys.path.append('../general')
from metric import get_metric, get_benefit
sys.path.append('../simple_regression')
from test_regression import get_positions, make_features


def get_deault_and_uniform_metric(pool, default_model, uniorm_model):
    train_pool, test_pool = pool.train_test_split()

    default_model.fit(
        make_features(train_pool.features, train_pool.positions), train_pool.targets
    )
    uniorm_model.fit(
        make_features(train_pool.features, train_pool.positions),
        train_pool.targets,
        sample_weight=(1 / train_pool.probas)
    )
    default_positions = get_positions(test_pool.features, default_model)
    uniform_positions = get_positions(test_pool.features, uniorm_model)

    return (
        get_metric(test_pool, default_positions),
        get_metric(test_pool, uniform_positions),
        get_benefit(test_pool),
        get_benefit(test_pool, 1 / test_pool.probas)
    )


def test_uniform_data_agregation(pool, default_models, uniorm_models):
    default_scores, uniorm_scores, default_benefits, uniform_benefits = [], [], [], []
    for default_model, uniorm_model in zip(default_models, uniorm_models):
        default_score, uniorm_score, default_benefit, uniorm_benefit = get_deault_and_uniform_metric(
            pool, default_model, uniorm_model
        )
        default_scores.append(default_score)
        uniorm_scores.append(uniorm_score)
        default_benefits.append(default_benefit)
        uniform_benefits.append(uniorm_benefit)
    return default_scores, uniorm_scores, default_benefits, uniform_benefits
