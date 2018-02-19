import numpy as np


NONE_POSITION = 11
POSITION_VARIANTS = np.arange(1, NONE_POSITION)


def extend_features(features, positions):
    repeated_input_positions = np.repeat(np.reshape(positions, (-1, 1)), len(POSITION_VARIANTS), axis=1)
    repeated_positions_variants = np.repeat([np.arange(1, NONE_POSITION)], len(features), axis=0)
    positions_one_hot = repeated_input_positions == repeated_positions_variants
    return np.concatenate((features, positions_one_hot), axis=1)


def predict_positions(features, score_model):
    repeated_features = np.repeat(features, len(POSITION_VARIANTS), axis=0)
    all_position_variants = np.repeat([POSITION_VARIANTS], len(features), axis=0).flatten()
    features_to_predict = extend_features(repeated_features, all_position_variants)
    scores = score_model.predict(features_to_predict)
    scores = np.reshape(scores, (len(features), len(POSITION_VARIANTS)))
    positions = np.argmax(scores, axis=1) + 1
    return positions
