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

    def _score_action(self, action, query=None, context=None, verbose=False, **kwargs):
        assert query is not None

        def score_by_query(other_query):
            if context is None:
                return self._bandits_by_queries[other_query]._score_action(action, verbose=verbose, **kwargs)
            else:
                return self._bandits_by_queries[other_query]._score_action(action, context, verbose=verbose, **kwargs)
        score = sum(
            score_by_query(near_query) * get_similarity(query, near_query)
            for near_query in self._nearest_queries[query]
        )
        if verbose:
            print(" action {} score: {}".format(action, score))
        return score

    def get_statistic_size(self, query):
        return self._statistic[query]

    def compile_rewards(self):
        for i, query in enumerate(self._bandits_by_queries):
            if i % 100 == 0:
                print(" {} / {} - {} %".format(
                    i,
                    len(self._bandits_by_queries),
                    round(100 * i / len(self._bandits_by_queries))
                ))
            self._bandits_by_queries[query].compile_rewards()

    def take_reward(self, query, action, reward, context=None):
        """
        action: int from self._actions
        reward: 0 or 1
        """
        assert reward in [0, 1]
        self._statistic[query] += 1
        if context is None:
            self._bandits_by_queries[query].take_reward(action, reward)
        else:
            self._bandits_by_queries[query].take_reward(action, reward, context)
