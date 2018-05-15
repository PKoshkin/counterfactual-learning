import numpy as np


def calculate_metric(answers_positions, target_positions, target, target_probs, prod_proba=100.0):
    mask = target_probs != 0
    target_probs[target_probs == 1.0] = prod_proba
    return np.mean(
        target[mask] / target_probs[mask] * (answers_positions[mask] == target_positions[mask])
    )
