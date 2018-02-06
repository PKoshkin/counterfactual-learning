import numpy as np

class QueryClusteringModelError(Exception):
    pass

# Модель: [features] + [position] -> score

class QueryClusteringModel:
    def __init__(self, models):
        self.models = models

    def fit(self, train_pools):
        if len(train_pools) != len(self.models):
            raise QueryClusteringModelError('Wrong pools noumber')

        for pool, model in zip(train_pools, self.models):
            features_with_positions = np.concatenate((pool.features, np.reshape(pool.positions, (-1, 1))), axis=1)
            model.fit(features_with_positions, pool.labels)

    def predict(self, test_pool):
        words = {word for query in test_pool.queries for word in query}
        words_wins = {word: [] for word in words}
        for label, query in zip(test_pool.labels, test_pool.queries):
            for word in query:
                words_wins[word].append(label)
        word_avarage_wins = {
            word: np.mean(words_wins[word])
            for word in words_wins
        }

        percentilies = np.linspace(0, 100, len(self.models) + 1)[1:]
        split_values = [np.percentile(list(word_avarage_wins.values()), percentile) for percentile in percentilies]
        result = []
        for feature, queries in zip(test_pool.features, test_pool.queries):
            average_win = np.mean([word_avarage_wins[word] for word in queries])
            index = 0
            for value in split_values:
                if average_win <= value:
                    break
                index += 1
            
            result.append(np.argmax([
                self.models[index].predict(list(feature) + [position])
                for position in test_pool.POSITIONS
            ]))

        return [position if position != 10 else 100 for position in result]
