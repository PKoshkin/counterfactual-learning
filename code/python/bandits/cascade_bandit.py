from bandit import Bandit
from damerau_levenshtien import damerau_levenshtein_distance


def get_nearest_queries(query, bandits_by_queries, queries_by_word, distance):
    return [
        other_query
        for word in query
        for other_query in queries_by_word[word]
        if damerau_levenshtein_distance(other_query, query) <= distance
    ]


class CascadeBandit(Bandit):
    def __init__(self, bandits_by_queries, queries_by_word, nearests_propotion=0.5):
        actions = next(iter(bandits_by_queries.values()))._actions
        super().__init__(actions)
        self._bandits_by_queries = bandits_by_queries
        self._nearests_propotion = nearests_propotion
        self._queries_by_word = queries_by_word

    def _get_nearest_queries(self, query):
        return get_nearest_queries(
            query,
            self._bandits_by_queries,
            self._queries_by_word,
            int(self._nearests_propotion * len(query))
        )

    def _get_weight(self, distance, max_distance):
        return (max_distance - distance + 1)  # / (max_distance + 1) / (max_distance / 2 + 1)

    def get_action(self, query, verbose=False):
        best_score = -1e16
        best_action = -1
        for action in self._actions:
            score = sum(
                self._bandits_by_queries[neighboor_query].score_action(action) * self._get_weight(
                    damerau_levenshtein_distance(query, neighboor_query),
                    int(self._nearests_propotion * len(query))
                ) for neighboor_query in self._get_nearest_queries(query)
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
        return sum(
            self._bandits_by_queries[neighboor_query].get_statistic_size()
            for neighboor_query in self._get_nearest_queries(query)
        )

    def take_reward(self, query, action, reward):
        """
        action: int from self._actions
        reward: 0 or 1
        """
        assert reward in [0, 1]
        self._bandits_by_queries[query].take_reward(action, reward)
