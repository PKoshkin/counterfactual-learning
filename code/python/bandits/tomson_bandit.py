from bandit import Bandit
from scipy.stats import beta
import numpy as np


class TomsonBandit(Bandit):
    def __init__(self, actions, init_params=[1, 1]):
        super().__init__(actions)
        self._init_params = init_params
        self._reward_distributions_params = {
            action: [init_params[0], init_params[1]]
            for action in self._actions
        }

    @staticmethod
    def _sample(params):
        """
        params: [losses, wins]
            losses: int
            wins: int
        """
        # beta "a" param equals to wins
        # beta "b" param equals to losses
        return beta.rvs(a=params[1], b=params[0], size=1)

    def get_action(self):
        rewards = [
            TomsonBandit._sample(self._reward_distributions_params[action])
            for action in self._actions
        ]
        return self._actions[np.argmax(rewards)]

    def score_action(self, action):
        return self._reward_distributions_params[action][1] - self._reward_distributions_params[action][0]

    def get_statistic_size(self):
        return sum([
            sum(self._reward_distributions_params[action]) - sum(self._init_params)
            for action in self._actions
        ])

    def get_greedy_action(self, statistic_size_threshold):
        wins = [
            self._reward_distributions_params[action][1]
            for action in self._actions
        ]
        if self.get_statistic_size() < statistic_size_threshold:
            return 4
        else:
            return self._actions[np.argmax(wins)]

    def take_reward(self, action, reward):
        """
        action: element from self._actions
        reward: 0 or 1
        """
        assert reward in [0, 1]
        self._reward_distributions_params[action][reward] += 1
