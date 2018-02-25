import numpy as np
from json import loads as json_from_string
from sklearn.model_selection import train_test_split
from constants import POSITION_VARIANTS, NONE_POSITION


class PoolError(Exception):
    pass


class Pool:
    def __init__(self, *args):
        self.NUM_FEATURES = 1052
        self.fields = [
            'features', 'positions', 'probas', 'targets', 'queries', 'prod_positions',
            'classification_labels', 'features_with_positions', 'features_with_one_hot_positions'
        ]
        for field in self.fields:
            self.__dict__[field] = []
        if len(args) == 1:
            if type(args[0]) != str:
                raise PoolError("Wrong constructor arguments")
            else:
                with open(args[0]) as handler:
                    data = [json_from_string(line) for line in handler]
                for line in data:
                    if line['images_metric'] is None:
                        position = NONE_POSITION
                        target = 0
                    else:
                        position = int(min(line['images_metric'][0], NONE_POSITION))
                        target = line['images_metric'][2] - line['images_metric'][1]
                    self.positions.append(position)
                    self.targets.append(target)
                    self.features.append(line['factors'][:self.NUM_FEATURES])
                    self.probas.append(line['p'])
                    self.queries.append(list(map(int, line['query'].split(' '))))
                    self.prod_positions.append(int(line['prod_pos']))
                    self.classification_labels.append(
                        0 if target < 0 else
                        1 if target == 0 else
                        2
                    )
                self.set_features(self.features)

            for field in self.fields:
                self.__dict__[field] = np.array(self.__dict__[field])

    def set_features(self, features):
        self.features = features
        positions_one_hot = np.array([
            [0 if position != current_position else 1 for position in POSITION_VARIANTS]
            for current_position in self.positions
        ])
        self.features_with_one_hot_positions = np.concatenate((self.features, positions_one_hot), axis=1)
        self.features_with_positions = np.concatenate(
            (self.features, np.reshape(self.positions, (-1, 1))),
            axis=1
        )

    def get_average_target(self, position):
        return np.mean(self.targets[self.positions == position])

    def log_features(self):
        self.features = np.log(1 + np.absolute(self.features))

    def set(self, *args):
        for field, value in zip(self.fields, args):
            self.__dict__[field] = value

    def split_by_position(self):
        pools = [Pool() for position in POSITION_VARIANTS]

        for i, position in enumerate(self.positions):
            for field in self.fields:
                position_variant_index = min(position, len(pools) - 1)
                pools[position_variant_index].__dict__[field].append(self.__dict__[field][i])

        for pool in pools:
            for field in self.fields:
                pool.__dict__[field] = np.array(pool.__dict__[field])

        return pools

    def train_test_split(self, test_size=0.3, need_indicies=False):
        indecies = np.arange(len(self.features))
        train_indecies, test_indecies = train_test_split(
            indecies, test_size=test_size, shuffle=True
        )
        train_fields_dict = {}
        test_fields_dict = {}
        for field in self.fields:
            train_fields_dict[field] = self.__dict__[field][train_indecies]
            test_fields_dict[field] = self.__dict__[field][test_indecies]

        test_pool, train_pool = Pool(), Pool()
        train_pool.set(*list(train_fields_dict.values()))
        test_pool.set(*list(test_fields_dict.values()))

        if not need_indicies:
            return train_pool, test_pool
        else:
            return train_pool, test_pool, train_indecies, test_indecies

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

    def calibrate(self, propotion=1):
        non_zero_counter = len(self.targets[self.targets != 0])
        zero_targets_cum = np.cumsum(self.targets == 0)
        last_needed_target_index = len(zero_targets_cum[
            zero_targets_cum <= (propotion * non_zero_counter)
        ])
        mask = np.concatenate((
                np.ones(last_needed_target_index).astype(bool),
                self.targets[last_needed_target_index:] != 0
            ), axis=0
        )
        for field in self.fields:
            self.__dict__[field] = self.__dict__[field][mask]

    def filter(self, mask):
        for field in self.fields:
            self.__dict__[field] = self.__dict__[field][mask]

    def copy(self):
        result = Pool()
        result.set(*[self.__dict__[field] for field in self.fields])
        return result
