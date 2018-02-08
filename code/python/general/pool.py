import numpy as np
from json import loads as json_from_string
from sklearn.model_selection import train_test_split

class PoolError(Exception):
    pass

class Pool:
    def __init__(self, *args):
        self.POSITIONS = list(range(10)) + [100]
        self.fields = ['features', 'positions', 'probas', 'targets', 'queries', 'prod_positions', 'classification_labels']
        if len(args) == 0:
            for field in self.fields:
                self.__dict__[field] = []
        elif len(args) == 1:
            if type(args[0]) != str:
                raise PoolError("Wrong constructor arguments")
            else:
                with open(args[0]) as handler:
                    data = [json_from_string(line) for line in handler]
                self.features = np.array([line['factors'][:1052] for line in data])
                self.positions = np.array([
                    int(line['images_metric'][0]) if line['images_metric'] is not None else 100
                    for line in data
                ])
                self.probas = np.array([line['p'] for line in data])
                self.targets = np.array([
                    (line['images_metric'][2] - line['images_metric'][1])
                    if line['images_metric'] is not None else 0
                    for line in data
                ])
                self.queries = [
                    list(map(int, line['query'].split(' ')))
                    for line in data
                ]
                self.prod_positions = [
                    int(line['prod_pos'])
                    for line in data
                ]
                self.classification_labels = [
                    (1 if (line['images_metric'][2] - line['images_metric'][1]) > 0 else 0)
                    if line['images_metric'] is not None else 0
                    for line in data
                ]

    def set(self, *args):
        for field, value in zip(self.fields, args):
            self.__dict__[field] = value

    def split_by_position(self):
        pools = [Pool() for position in self.POSITIONS]

        for i in range(len(self.features)):
            index = self.position[i] if self.position[i] in list(range(10)) else 10
            for field in self.fields:
                pools[index].__dict__[field].append(self.__dict__[field][i])

        for pool in pools:
            for field in self.fields:
                pool.__dict__[field] = np.array(pool.__dict__[field])

        return pools

    def train_test_split(self, test_size=0.3):
        features_train, features_test,\
        positions_train, positions_test,\
        targets_train, targets_test,\
        proba_train, proba_test,\
        queries_train, queries_test,\
        prod_positions_train, prod_positions_test,\
        classification_labels_train, classification_labels_test = train_test_split(
            self.features, self.positions, self.targets, self.probas,
            self.queries, self.prod_positions, self.classification_labels,
            test_size=test_size, shuffle=True
        )

        test_pool, train_pool = Pool(), Pool()
        test_pool.set(features_test, positions_test, proba_test, targets_test, queries_test, prod_positions_test, classification_labels_test)
        train_pool.set(features_train, positions_train, proba_train, targets_train, queries_train, prod_positions_train, classification_labels_train)

        return train_pool, test_pool

    def split_by_queries(self):
        POOLS_NUMBER = 3
        words = {word for query in self.queries for word in query}
        words_wins = {word: [] for word in words}
        for target, query in zip(self.targets, self.queries):
            for word in query:
                words_wins[word].append(target)
        word_avarage_wins = {
            word: np.mean(words_wins[word])
            for word in words_wins
        }

        pools = [Pool() for i in range(POOLS_NUMBER)]

        for i in range(len(self.features)):
            average_win = np.mean([word_avarage_wins[word] for word in self.queries[i]])
            index = 0 if average_win < 0 else 1 if average_win == 0 else 2
            for field in self.fields:
                pools[index].__dict__[field].append(self.__dict__[field][i])

        for pool in pools:
            for field in self.fields:
                pool.__dict__[field] = np.array(pool.__dict__[field])

        return pools
