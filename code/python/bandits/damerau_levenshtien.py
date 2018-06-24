import numpy as np


INSERT_COST = 1
DELETE_COST = 1
REPLACE_COST = 1
INF = 1e16


def levenshtein_distance(first_str, second_str):
    d = np.zeros([len(first_str) + 1, len(second_str) + 1])
    d[0, 0] = 0
    for j in range(1, len(second_str) + 1):
        d[0, j] = d[0, j - 1] + INSERT_COST

    for i in range(1, len(first_str) + 1):
        d[i, 0] = d[i - 1, 0] + DELETE_COST
        for j in range(1, len(second_str) + 1):
            d[i, j] = min(
                d[i - 1, j] + DELETE_COST,
                d[i, j - 1] + INSERT_COST,
                d[i - 1, j - 1] + (REPLACE_COST if first_str[i - 1] != second_str[j - 1] else 0)
            )
    return d[-1, -1]


def damerau_levenshtein_distance(first_str, second_str):
    d = np.zeros([len(first_str) + 1, len(second_str) + 1])
    d[0, 0] = 0
    for j in range(1, len(second_str) + 1):
        d[0, j] = d[0, j - 1] + INSERT_COST

    for i in range(1, len(first_str) + 1):
        d[i, 0] = d[i - 1, 0] + DELETE_COST
        for j in range(1, len(second_str) + 1):
            d[i, j] = min(
                d[i - 1, j] + DELETE_COST,
                d[i, j - 1] + INSERT_COST,
                d[i - 1, j - 1] + (REPLACE_COST if first_str[i - 1] != second_str[j - 1] else 0),
                INF if i < 2 or j < 2 else (
                    1 if first_str[i - 1] == second_str[j - 2] and first_str[i - 2] == second_str[j - 1] else INF
                )
            )
    print(d)
    return d[-1, -1]
