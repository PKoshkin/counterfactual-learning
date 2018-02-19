import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

from math import ceil
from collections import defaultdict
from frozendict import frozendict


ALGO_KEY = 'algo'
STRATEGY_KEY = 'strategy'
BATCH_SIZE_KEY = 'batch size'
INITIAL_SIZE_KEY = 'initial size'
MAX_QUERIES_KEY = 'max queries'
START_LINE = "Algorithm with following features was applied:"


class IncorrectParams(Exception):
    def __init__(self, keys, *args, **kwargs):
        discribtion = "There is no either " + " either ".join(keys) + " in params"
        super().__init__(discribtion, *args, **kwargs)


def _get_params(results_lines, line_ind):
    algo_params = {}

    while results_lines[line_ind] != "":
        key, value = results_lines[line_ind].split(': ')
        if value.isdigit():
            algo_params[key] = int(value)
        else:
            algo_params[key] = value
        line_ind += 1
    return line_ind, frozendict(algo_params)


def _get_batches_num(params):
    try:
        return ceil((params[MAX_QUERIES_KEY] - params[INITIAL_SIZE_KEY]) / params[BATCH_SIZE_KEY])
    except Exception as e:
        raise IncorrectParams([MAX_QUERIES_KEY, INITIAL_SIZE_KEY, BATCH_SIZE_KEY])


def _get_name(params):
    try:
        return params[ALGO_KEY] + " " + params[STRATEGY_KEY]
    except Exception as e:
        raise IncorrectParams([ALGO_KEY, STRATEGY_KEY])


def _skip_blank(results_lines, line_ind):
    while line_ind < len(results_lines) and results_lines[line_ind] == "":
        line_ind += 1
    return line_ind


def read_results(filename):
    results = defaultdict(list)
    prev_params = frozendict()

    results_lines = [line.strip() for line in open(filename)]

    line_ind = _skip_blank(results_lines, 0)
    while line_ind < len(results_lines):
        if results_lines[line_ind] == START_LINE:
            line_ind += 1
        line_ind, algo_params = _get_params(results_lines, line_ind)
        if algo_params != prev_params:
            prev_params = algo_params
            results[algo_params].append([])
        line_ind = _skip_blank(results_lines, line_ind)

        results[algo_params][-1].append([])
        while results_lines[line_ind] != "":
            results[algo_params][-1][-1].append(float(results_lines[line_ind]))
            line_ind += 1
        line_ind = _skip_blank(results_lines, line_ind)

    for key, value in results.items():
        results[key] = np.array(value)
    return results


def draw_plots(results):
    for algo_params, metrics in results.items():
        plt.errorbar(
            np.arange(_get_batches_num(algo_params)),
            metrics[-1].mean(axis=0),
            yerr=metrics[-1].var(axis=0),
            label=_get_name(algo_params),
        )
    plt.legend()
    plt.show()


def mannwhitneyu_test(results, key_a, key_b, pvalue_bound=0.05):
    a = results[key_a][-1, :, -1]
    b = results[key_b][-1, :, -1]
    _, pvalue = ss.mannwhitneyu(a, b, alternative='less')
    if pvalue < pvalue_bound:
        verdict = " is less than "
    else:
        _, pvalue = ss.mannwhitneyu(a, b, alternative='greater')
        if pvalue < pvalue_bound:
            verdict = ' is greater than '
        else:
            _, pvalue = ss.mannwhitneyu(a, b, alternative='two-sided')
            verdict = ' is equal to '

    print(_get_name(key_a) + verdict + _get_name(key_b) + " with pvalue " + str(pvalue))


def print_stats(results):
    for key, value in results.items():
        print(
            _get_name(key) + ' results:\n'
            'Mean: ' + str(value[-1, :, -1].mean()) + '\n' +
            'Std: ' + str(value[-1, :, -1].std(ddof=1)) + '\n'
        )


__all__ = [read_results, draw_plots, mannwhitneyu_test, print_stats]
