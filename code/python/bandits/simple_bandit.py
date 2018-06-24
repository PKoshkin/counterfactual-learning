from bandit import Bandit
import numpy as np


class SimpleBandit(Bandit):
    def __init__(self, actions, memory_size=1000000):
        super().__init__(actions)
        self._memory_size = memory_size
        self._memory = {
            action: [] for action in actions
        }

    def get_action(self):
        non_empty_actions = list(filter(lambda action: len(self._memory[action]) != 0, self._actions))
        if len(non_empty_actions) == 0:
            return np.random.choice(self._actions)
        mean_rewards = {
            action: np.mean(self._memory[action])
            for action in non_empty_actions
        }
        return max(non_empty_actions, key=lambda action: mean_rewards[action])

    def take_reward(self, action, reward):
        self._memory[action].append(reward)
        if len(self._memory[action]) > self._memory_size:
            self._memory[action] = self._memory[action][1:]
