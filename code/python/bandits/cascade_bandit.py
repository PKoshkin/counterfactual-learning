from collections import defaultdict
from bandit import Bandit


def get_similarity(query_1, query_2):
    return len(set(query_1) & set(query_2)) / len(set(query_1) | set(query_2))


class CascadeBandit(Bandit):
    def __init__(self, bandits_by_queries, nearest_queries, nearests_propotion=0.5):
        actions = next(iter(bandits_by_queries.values()))._actions
        super().__init__(actions)
        self._bandits_by_queries = bandits_by_queries
        self._nearests_propotion = nearests_propotion
        self._nearest_queries = nearest_queries
        self._statistic = defaultdict(lambda: 0)

    def get_action(self, query, verbose=False):
        best_score = -1e16
        best_action = -1
        for action in self._actions:
            score = sum(
                self._bandits_by_queries[near_query].score_action(action) * get_similarity(query, near_query)
                for near_query in self._nearest_queries[query]
            )
            if verbose:
                print(score)
            if score > best_score:
                best_score = score
                best_action = action
        if verbose:
            print(" best score: {}".format(best_score))
        return best_action

    def get_statistic_size(self, query):
        return self._statistic[query]

    def take_reward(self, query, action, reward):
        """
        action: int from self._actions
        reward: 0 or 1
        """
        assert reward in [0, 1]
        self._statistic[query] += 1
        self._bandits_by_queries[query].take_reward(action, reward)
