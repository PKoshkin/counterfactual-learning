import numpy as np


class Bandit:
    def __init__(self, actions):
        self._actions = actions

    def _score_action(self, action, **kwargs):
        pass

    def get_action(self, **kwargs):
        return self._actions[np.argmax([
            self._score_action(action, **kwargs)
            for action in self._actions
        ])]

    def take_reward(self, action, reward):
        pass
