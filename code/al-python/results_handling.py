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
TRAIN_SIZE_KEY = 'train pool size'
TEST_SIZE_KEY = 'test pool size'

ERROR_LINE = "error!!!"
START_LINE = "Algorithm with following features was applied:"

STRATEGY_COLORS = {
    "US": 'blue',
    "random": 'orange',
    "diversity": 'green',
}


class IncorrectParams(Exception):
    def __init__(self, keys, *args, **kwargs):
        discribtion = "There is no either " + " either ".join(keys) + " in params"
        super().__init__(discribtion, *args, **kwargs)


def _get_color(name):
    for key, color in STRATEGY_COLORS.items():
        if name.startswith(key):
            return color
    return None


def _get_params(results_lines, line_ind):
    algo_params = {}

    while results_lines[line_ind] != "":
        print(results_lines[line_ind])
        splited = results_lines[line_ind].split(': ')
        key = splited[0]
        value = ': '.join(splited[1:])
        if value.isdigit():
            algo_params[key] = int(value)
        else:
            algo_params[key] = value
        line_ind += 1
    return line_ind, frozendict(algo_params)


def _get_batches_num(params):
    try:
        tmp = float(params[MAX_QUERIES_KEY] - params[INITIAL_SIZE_KEY]) / params[BATCH_SIZE_KEY]
        return int(ceil(tmp))
    except Exception as e:
        raise IncorrectParams([MAX_QUERIES_KEY, INITIAL_SIZE_KEY, BATCH_SIZE_KEY])


def _get_name(params, keys=None):
    if keys is None:
        keys = [STRATEGY_KEY]
    try:
        return " ".join([
            (key + '_' if key != STRATEGY_KEY else "") + str(params[key])
            for key in keys
        ])
    except Exception as e:
        raise IncorrectParams(keys)


def _skip_blank(results_lines, line_ind):
    while line_ind < len(results_lines) and results_lines[line_ind] == "":
        line_ind += 1
    return line_ind


def parse_result_file(filename):
    with open(filename) as results_file:
        results_lines = [line.strip() for line in open(filename)]

    line_ind = _skip_blank(results_lines, 0)
    if results_lines[line_ind] == START_LINE:
        line_ind += 1
    line_ind = _skip_blank(results_lines, line_ind)
    line_ind, algo_params = _get_params(results_lines, line_ind)
    line_ind = _skip_blank(results_lines, line_ind)

    results = []

    while line_ind < len(results_lines) and results_lines[line_ind] != "":
        results.append(float(results_lines[line_ind]))
        line_ind += 1

    return algo_params, np.array(results)


def get_plot_data(filenames, result_keys_for_name=None):
    results = {}
    for filename in filenames:
        params, result = parse_result_file(filename)
        name = _get_name(params, result_keys_for_name)
        if name not in results:
            results[name] = np.array([result])
        else:
            results[name] = np.concatenate([[result], results[name]], axis=0)

    plot_data = {}
    for name, metrics in results.items():
        x_start = float(params[INITIAL_SIZE_KEY]) / params[TRAIN_SIZE_KEY]
        x_end = float(params[MAX_QUERIES_KEY]) / params[TRAIN_SIZE_KEY]
        x = np.linspace(x_start, x_end, len(metrics))

        plot_data[name] = {
            'x': x,
            'mean': np.mean(metrics, axis=0),
            'std': np.std(metrics, axis=0) / np.sqrt(len(metrics))
        }

    return plot_data


def draw_plots(
        plot_data,
        fontsize=10,
        thick_num=None,
        title="Active learning algorithms' performace",
        xlabel="Training instances share",
        ylabel="Relevance",
        image_path=None,
        *args,
        **kwargs):
    for name, data in plot_data.items():
        if thick_num is None:
            errorevery = 1
        else:
            errorevery = len(data['x']) // thick_num

        plt.errorbar(
            data['x'],
            data['mean'],
            color=_get_color(name),
            yerr=data['std'],
            label=name,
            errorevery=errorevery,
            *args, **kwargs
        )

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend()
    if image_path is None:
        plt.show()
    else:
        plt.savefig(image_path, bbox_inches='tight')


def mannwhitneyu_test(results, key_a, key_b, pvalue_bound=0.05):
    a = results[key_a][-1][:, -1]
    b = results[key_b][-1][:, -1]
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


def print_stats(results, keys=None):
    if keys is None:
        keys = results.keys()
    for key in keys:
        value = results[key]
        print(
            _get_name(key) + ' results:\n'
            'Tests num: ' + str(value[-1].shape[0]) + '\n' +
            'Mean: ' + str(value[-1][:, -1].mean()) + '\n' +
            'Std: ' + str(value[-1][:, -1].std(ddof=1)) + '\n'
        )
