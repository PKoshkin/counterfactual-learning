import numpy as np

def metric(answers_positions, target_positions, target, target_probs):
    return np.mean(
        target / target_probs * (answers_positions == target_positions)
    )
