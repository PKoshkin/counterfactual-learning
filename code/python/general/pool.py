import numpy as np
from json import loads as json_from_string
from sklearn.model_selection import train_test_split


class PoolError(Exception):
    pass


class Pool:
    def __init__(self, *args):
        self.NONE_POSITION = 10
        self.NUM_FEATURES = 1052
        self.POSITIONS = np.arange(self.NONE_POSITION)
        self.fields = [
            'features', 'positions', 'probas', 'targets', 'queries', 'prod_positions',
            'classification_labels', 'regression_features', 'regression_prediction', 'positions_variants'
        ]
        if len(args) == 0:
            for field in self.fields:
                self.__dict__[field] = []
        elif len(args) == 1:
            if type(args[0]) != str:
                raise PoolError("Wrong constructor arguments")
            else:
                with open(args[0]) as handler:
                    data = [json_from_string(line) for line in handler]
                self.features = np.array([line['factors'][:self.NUM_FEATURES] for line in data])
                self.positions = np.clip([
                    int(line['images_metric'][0]) if line['images_metric'] is not None else self.NONE_POSITION
                    for line in data
                ], -1, self.NONE_POSITION)
                self.probas = np.array([line['p'] for line in data])
                self.targets = np.array([
                    (line['images_metric'][2] - line['images_metric'][1])
                    if line['images_metric'] is not None else 0
                    for line in data
                ])
                self.queries = np.array([
                    list(map(int, line['query'].split(' ')))
                    for line in data
                ])
                self.prod_positions = np.array([
                    int(line['prod_pos'])
                    for line in data
                ])
                self.classification_labels = np.array([(
                        0 if (line['images_metric'][2] - line['images_metric'][1]) < 0 else
                        1 if (line['images_metric'][2] - line['images_metric'][1]) == 0 else
                        2
                    ) if line['images_metric'] is not None else 1
                    for line in data
                ])
                positions_one_hot = np.array([
                    [0 if position != current_position else 1 for position in self.POSITIONS]
                    for current_position in self.positions
                ])
                self.regression_features = np.concatenate((self.features, positions_one_hot), axis=1)
                self.regression_prediction = np.reshape(self.targets, (-1, 1))
                self.positions_variants = np.array([(
                        0 if (line['images_metric'][2] - line['images_metric'][1]) < 0 else
                        1 if (line['images_metric'][2] - line['images_metric'][1]) == 0 else
                        2
                    ) if line['images_metric'] is not None else 3
                    for line in data
                ])

    def log_features(self):
        self.features = np.log(1 + np.absolute(self.features))

    def set(self, *args):
        for field, value in zip(self.fields, args):
            self.__dict__[field] = value

    def split_by_position(self):
        pools = [Pool() for position in self.POSITIONS]

        for i, position in enumerate(self.positions):
            for field in self.fields:
                if position != self.NONE_POSITION:
                    pools[position].__dict__[field].append(self.__dict__[field][i])
                else:
                    for tmp_position in self.POSITIONS:
                        pools[tmp_position].__dict__[field].append(self.__dict__[field][i])

        for pool in pools:
            for field in self.fields:
                pool.__dict__[field] = np.array(pool.__dict__[field])

        return pools

    def train_test_split(self, test_size=0.3):
        train_fields_dict = {}
        test_fields_dict = {}
        for field in self.fields:
            train_fields_dict[field], test_fields_dict[field] = train_test_split(
                self.__dict__[field], test_size=test_size, shuffle=True
            )

        test_pool, train_pool = Pool(), Pool()
        train_pool.set(*list(train_fields_dict.values()))
        test_pool.set(*list(test_fields_dict.values()))

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
