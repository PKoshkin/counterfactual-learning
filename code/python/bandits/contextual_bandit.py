from bandit import Bandit
from catboost import CatBoostClassifier
import sys
sys.path.append("../utils")
from log import log


class ContextualBandit(Bandit):
    def __init__(self, actions, init_params=[1, 1]):
        super().__init__(actions)
        self._init_params = init_params
        self._reward_distributions_params = {
            action: [init_params[0], init_params[1]]
            for action in self._actions
        }
        self._model_fited = False
        self._statistic = 0
        self._model = CatBoostClassifier()
        self._features = []
        self._rewards = []

    def _is_fitable(self):
        return len(set(self._rewards)) > 1

    def _score_action_stat(self, action, verbose=False):
        wins = self._reward_distributions_params[action][1]
        losses = self._reward_distributions_params[action][0]
        assert wins + losses > 0
        score = (wins - losses) / (wins + losses)
        if verbose:
            log("action {} stat score is {}".format(action, score))
        return score

    def _score_action_model(self, action, context, verbose=False):
        score = self._model.predict_proba([[action] + context])[0, 1]
        if verbose:
            log("action {} model score is {}".format(action, score))
        return score

    def _score_action(self, action, context=None, statistic_threshold=0, verbose=False):
        if context is None:
            return self._score_action_stat(action, verbose=verbose)

        if self._statistic < statistic_threshold or not self._is_fitable():
            return self._score_action_stat(action, verbose=verbose)
        else:
            if not self._model_fited:
                if verbose:
                    log("fitting model...")
                self._model.fit(self._features, self._rewards, verbose=False)
                self._model_fited = True
                if verbose:
                    log("model fited.")
            return self._score_action_model(action, context, verbose=verbose)

    def get_statistic_size(self):
        return self._statistic

    def compile_rewards(self):
        if self._is_fitable():
            self._model.fit(self._features, self._rewards, verbose=False)
            self._model_fited = True

    def take_reward(self, action, reward, context):
        """
        action: element from self._actions
        reward: 0 or 1
        """
        assert reward in [0, 1]
        self._statistic += 1
        self._features.append([action] + context)
        self._rewards.append(reward)
        self._reward_distributions_params[action][reward] += 1
