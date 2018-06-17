import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import argparse
import json

from math import ceil
from collections import defaultdict
from frozendict import frozendict

from strategies import STRATEGIES


RANDOM_NAME = 'random'

ALGO_KEY = 'algo'
STRATEGY_KEY = 'strategy'
BATCH_SIZE_KEY = 'batch size'
INITIAL_SIZE_KEY = 'initial size'
MAX_QUERIES_KEY = 'max queries'
TRAIN_SIZE_KEY = 'train pool size'
TEST_SIZE_KEY = 'test pool size'
PARAMS_KEY = 'strategy_params'

ERROR_LINE = "error!!!"
START_LINE = "Algorithm with following features was applied:"
BEGIN_BEST_ITER = 'best iteration'
BEGIN_FINAL_METRIC = 'metric on test using classifier from best iteration'

STRATEGY_COLORS = {
    'US': 'blue',
    'random': 'orange',
    'diversity': 'lightgreen',
    'US-density': 'pink',
    'density': 'black',
    'diversity-density': 'red',
    'random_part_0.1': 'red',
    'random_part_0.3': 'blue',
    'random_part_0.5': 'green',
}


class IncorrectParams(Exception):
    def __init__(self, keys, *args, **kwargs):
        discribtion = "There is no either " + " either ".join(keys) + " in params"
        super().__init__(discribtion, *args, **kwargs)


def _get_color(name):
    return STRATEGY_COLORS.get(name.split('_')[0])


def _get_params(results_lines, line_ind):
    algo_params = {}

    while results_lines[line_ind] != "":
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
    result = params[STRATEGY_KEY]
    if keys is not None:
        strategy_params = json.loads(params[PARAMS_KEY])
        for param, param_value in strategy_params.items():
            if param in STRATEGIES:
                strategy = param
                curr_strategy_params = param_value
                for param, param_value in curr_strategy_params.items():
                    if param in keys:
                        result += '.' + strategy + '_' + param + '_' + str(param_value)
            else:
                if param in keys:
                    result += '.' + param + '_' + str(param_value)

    return result


def _skip_blank(results_lines, line_ind):
    while line_ind < len(results_lines) and results_lines[line_ind] == "":
        line_ind += 1
    return line_ind


def parse_result_block(results_lines, line_ind):
    if results_lines[line_ind] == START_LINE:
        line_ind += 1
    line_ind = _skip_blank(results_lines, line_ind)
    line_ind, algo_params = _get_params(results_lines, line_ind)
    line_ind = _skip_blank(results_lines, line_ind)

    results = []

    while not results_lines[line_ind].startswith(BEGIN_BEST_ITER):
        results.append(float(results_lines[line_ind]))
        line_ind += 1

    assert results_lines[line_ind].startswith(BEGIN_BEST_ITER)
    best_iter = int(results_lines[line_ind].split(': ')[1])
    line_ind += 1

    assert results_lines[line_ind].startswith(BEGIN_FINAL_METRIC)
    final_metric = float(results_lines[line_ind].split(': ')[1])
    line_ind += 1

    return algo_params, np.array(results), line_ind, best_iter, final_metric


def get_results(filename, strategies_to_data=None, result_keys_for_name=None):
    with open(filename) as results_file:
        results_lines = [line.strip() for line in open(filename)]

    results = {}
    line_ind = _skip_blank(results_lines, 0)

    while line_ind < len(results_lines):
        params, result, line_ind, best_iter, final_metric = parse_result_block(
            results_lines, line_ind
        )
        if strategies_to_data is not None and params[STRATEGY_KEY] not in strategies_to_data:
            continue
        name = _get_name(params, result_keys_for_name)
        if name not in results:
            results[name] = (np.array([result]), [final_metric], [best_iter])
        else:
            results[name][0] = np.concatenate([[result], results[name][0]], axis=0)
            results[name][1].append(final_metric)
            results[name][2].append(best_iter)

        line_ind = _skip_blank(results_lines, line_ind)

    return results, params


def get_plot_data(
        filename, result_keys_for_name=None, max_size_draw=None, strategies_to_data=None,
        results=None, params=None):
    if results is None or params is None:
        results, params = get_results(filename, strategies_to_data, result_keys_for_name)
    plot_data = {}
    for name, (metrics, _, _) in results.items():
        x_start = float(params[INITIAL_SIZE_KEY]) / params[TRAIN_SIZE_KEY]
        x_end = float(params[MAX_QUERIES_KEY]) / params[TRAIN_SIZE_KEY]
        x_batch = float(params[BATCH_SIZE_KEY]) / params[TRAIN_SIZE_KEY]
        if max_size_draw is not None:
            x_end = min(x_end, float(max_size_draw) / params[TRAIN_SIZE_KEY])
        x = np.arange(x_start, x_end, x_batch)

        plot_data[name] = {
            'x': x,
            'mean': np.mean(metrics, axis=0)[:len(x)],
            'std': np.std(metrics, axis=0)[:len(x)] / np.sqrt(len(metrics))
        }

    return plot_data


def _add_postfix(path, postfix):
    names = path.split('.')
    if len(names) == 1:
        name_ind = 0
    else:
        name_ind = -2
    names[name_ind] += '_' + postfix
    return '.'.join(names)


def _finilize_and_save(title, xlabel, ylabel, fontsize, image_path, postfix):
    title += ' ' + postfix
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend()
    if image_path is None:
        plt.show()
    else:
        image_path = _add_postfix(image_path, postfix)
        plt.savefig(image_path, bbox_inches='tight')


def _draw_plot(name, data, thick_num, *args, **kwargs):
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


def draw_plots(
        plot_data,
        compare_with_random,
        title,
        fontsize=10,
        thick_num=None,
        xlabel="Training instances share",
        ylabel="Relevance",
        image_path=None,
        *args,
        **kwargs):

    for name, data in plot_data.items():
        if compare_with_random and name == RANDOM_NAME:
            continue
        if compare_with_random:
            print(name)
            plt.figure()
        _draw_plot(name, data, thick_num, *args, **kwargs)
        if compare_with_random:
            _draw_plot(RANDOM_NAME, plot_data[RANDOM_NAME], thick_num, *args, **kwargs)
            _finilize_and_save(title, xlabel, ylabel, fontsize, image_path, name)

    if not compare_with_random:
        _finilize_and_save(title, xlabel, ylabel, fontsize, image_path, '')


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


def _add_common_params(parser):
    parser.add_argument('--image_path', default='plots.png')
    parser.add_argument('--key_for_name', nargs='*')
    parser.add_argument('--max_size_draw')
    parser.add_argument('--strategies_to_data', nargs='+')
    parser.add_argument('--title', default='Active learning algorithms\' perfomance')


def parse_args():
    parser = argparse.ArgumentParser()
    mode_adder = parser.add_subparsers(dest='mode')

    single_result_parser = mode_adder.add_parser('single')
    single_result_parser.add_argument('--results_path', default='results.txt')
    single_result_parser.add_argument('--compare_with_random', action='store_true')
    _add_common_params(single_result_parser)

    many_results_parser = mode_adder.add_parser('many')
    many_results_parser.add_argument('--results_paths', nargs='+', required=True)
    many_results_parser.add_argument('--names', nargs='+', required=True)
    _add_common_params(many_results_parser)

    args = parser.parse_args()
    if args.mode == 'many':
        if len(args.results_paths) != len(args.names):
            parser.error('results_paths number should be equal to names number')

    return args


def main():
    args = parse_args()

    if args.mode == 'single':
        plot_data = get_plot_data(
            args.results_path,
            args.key_for_name,
            max_size_draw=args.max_size_draw,
            strategies_to_data=args.strategies_to_data,
        )
        draw_plots(plot_data, args.compare_with_random, args.title, image_path=args.image_path)
    if args.mode == 'many':
        plot_datas = defaultdict(dict)
        for result_path, result_name in zip(args.results_paths, args.names):
            plot_data = get_plot_data(
                result_path,
                args.key_for_name,
                max_size_draw=args.max_size_draw,
                strategies_to_data=args.strategies_to_data,
            )
            print(result_name)
            for name, data in plot_data.items():
                print(name)
                plot_datas[name][result_name] = data

        for name, plot_data in plot_datas.items():
            title = args.title + ' ' + name
            draw_plots(plot_data, False, title, image_path=_add_postfix(args.image_path, name))


if __name__ == '__main__':
    main()
