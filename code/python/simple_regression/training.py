import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

def make_features(main_features, positions):
    return np.concatenate([main_features, np.reshape(positions, (-1, 1))], axis=1)

def get_positions(features_to_answer, model):
    position_variants = np.array(list(range(10)) + [100])
    answers = []
    for features in features_to_answer:
        features = np.repeat(np.reshape(features, (1, -1)), len(position_variants), axis=0)
        scores = model.predict(make_features(features, position_variants))
        answers.append(np.argmax(scores))
    return answers

def metric(answers_positions, target_positions, target, target_probs):
    return np.mean(
        target / target_probs * (answers_positions == target_positions)
    )

def validate_count(answers_positions, target_positions, target, target_probs):
    return np.mean(
            1 / target_probs * (answers_positions == target_positions)
    )

def get_metric(features, positions, labels, probas, model, validate=False):
    features_train, features_test,\
    positions_train, positions_test,\
    labels_train, labels_test,\
    proba_train, proba_test = train_test_split(
        features, positions, labels, probas, test_size=0.3, shuffle=True
    )
    model.fit(make_features(features_train, positions_train), labels_train)
    answers = get_positions(features_test, model)

    if validate:
        for j in range(10):
            constant_answers = [j for i in range(len(features_test))]
            print("j={}, validate: {}, metric: {}".format(
                j,
                validate_count(constant_answers, positions_test, labels_test, proba_test),
                metric(constant_answers, positions_test, labels_test, proba_test)
            ))

    return metric(answers, positions_test, labels_test, proba_test)
