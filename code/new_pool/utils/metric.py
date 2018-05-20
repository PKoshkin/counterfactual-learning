import numpy as np


def calculate_metric(answers_positions, target_positions, target, target_probs):
    # mask = (target_probs != 0) * (target_probs != 1.0)
    mask = (target_probs != 0)
    return np.mean(
        target[mask] / target_probs[mask] * (answers_positions[mask] == target_positions[mask])
    )
