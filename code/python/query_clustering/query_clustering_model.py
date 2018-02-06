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


        words_wins = {}
        for pool in train_pools:
            for label, query in zip(pool.labels, pool.queries):
                for word in query:
                    if word not in words_wins:
                        words_wins[word] = [label]
                    else:
                        words_wins[word].append(label)
        self.word_average_wins = {
            word: np.mean(words_wins[word])
            for word in words_wins
        }

        for pool, model in zip(train_pools, self.models):
            features_with_positions = np.concatenate((pool.features, np.reshape(pool.positions, (-1, 1))), axis=1)
            model.fit(features_with_positions, pool.labels)

    def predict(self, test_pool):
        result = []
        for feature, queries in zip(test_pool.features, test_pool.queries):
            average_win = 0
            counter = 0

            bad_counter = 0
            for word in queries:
                if word in self.word_average_wins:
                    average_win += self.word_average_wins[word]
                    counter += 1
                else:
                    bad_counter += 1

            if counter != 0:
                average_win /= counter

            index = 0 if average_win < 0 else 1 if average_win == 0 else 2
            
            result.append(np.argmax([
                self.models[index].predict(list(feature) + [position])
                for position in test_pool.POSITIONS
            ]))

        return [position if position != 10 else 100 for position in result]
