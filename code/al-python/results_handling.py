import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import argparse
import json
import sys

from math import ceil
from collections import defaultdict
from frozendict import frozendict

from strategies import STRATEGIES


RANDOM_NAME = 'random'
ALL_DATA = 'all_data'

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
START_VALIDATION_METRICS = 'Validation metrics:'
START_TEST_METRICS = 'Test metrics:'
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


def _get_mean_std(scores, cut_outliers):
    # cut bottom
    cutted_scores = np.partition(scores, cut_outliers)[cut_outliers:]
    # cut top
    cutted_scores = np.partition(cutted_scores, -cut_outliers)[:-cut_outliers]
    mean = np.mean(cutted_scores)
    std = np.std(cutted_scores) / np.sqrt(len(cutted_scores))
    return (mean, std)


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

    val_results = []
    test_results = []

    line_ind = _skip_blank(results_lines, line_ind)
    assert results_lines[line_ind] == START_VALIDATION_METRICS
    line_ind += 1
    while results_lines[line_ind] != "":
        val_results.append(float(results_lines[line_ind]))
        line_ind += 1

    line_ind = _skip_blank(results_lines, line_ind)
    assert results_lines[line_ind] == START_TEST_METRICS
    line_ind += 1
    while results_lines[line_ind] != "":
        test_results.append(float(results_lines[line_ind]))
        line_ind += 1

    return algo_params, np.array(test_results), np.array(val_results), line_ind


def get_results(filename, strategies_to_data=None, result_keys_for_name=None):
    with open(filename) as results_file:
        results_lines = [line.strip() for line in open(filename)]

    results = {}
    line_ind = _skip_blank(results_lines, 0)

    while line_ind < len(results_lines):
        params, val_result, test_result, line_ind = parse_result_block(
            results_lines, line_ind
        )
        if strategies_to_data is not None and params[STRATEGY_KEY] not in strategies_to_data:
            continue
        name = _get_name(params, result_keys_for_name)
        if name not in results:
            results[name] = [np.array([val_result]), np.array([test_result])]
        else:
            results[name][0] = np.concatenate([[val_result], results[name][0]], axis=0)
            results[name][1] = np.concatenate([[test_result], results[name][1]], axis=0)

        line_ind = _skip_blank(results_lines, line_ind)

    return results, params


def get_plot_data(
        filename, result_keys_for_name=None, max_size_draw=None, strategies_to_data=None, cut_outliers=3,
        results=None, params=None):
    if results is None or params is None:
        results, params = get_results(filename, strategies_to_data, result_keys_for_name)
    x_start = float(params[INITIAL_SIZE_KEY]) / params[TRAIN_SIZE_KEY]
    x_end = float(params[MAX_QUERIES_KEY]) / params[TRAIN_SIZE_KEY]
    x_batch = float(params[BATCH_SIZE_KEY]) / params[TRAIN_SIZE_KEY]
    iters_num = int(round((x_end - x_start) / x_batch)) + 1
    x = np.linspace(x_start, x_end, iters_num)

    plot_data = {}
    last = []
    for name, (metrics, _) in results.items():
        cutted_metrics = np.partition(metrics, cut_outliers, axis=0)[cut_outliers:]
        cutted_metrics = np.partition(metrics, -cut_outliers, axis=0)[:-cut_outliers]
        mean = np.mean(cutted_metrics, axis=0)
        if x_end == 1.0:
            last.append(metrics[:, -1])

        plot_data[name] = {
            'x': x,
            'mean': mean,
            'std': np.std(cutted_metrics, axis=0) / np.sqrt(len(cutted_metrics))
        }
    if x_end == 1.0:
        last_mean, last_std = _get_mean_std(np.concatenate(last), len(last) * cut_outliers)
        for name in results:
            plot_data[name]['mean'][-1] = last_mean
            plot_data[name]['std'][-1] = last_std

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
            plt.figure()
        _draw_plot(name, data, thick_num, *args, **kwargs)
        if compare_with_random:
            _draw_plot(RANDOM_NAME, plot_data[RANDOM_NAME], thick_num, *args, **kwargs)
            _finilize_and_save(title, xlabel, ylabel, fontsize, image_path, name)

    if not compare_with_random:
        _finilize_and_save(title, xlabel, ylabel, fontsize, image_path, '')


def print_test_scores(results, report_file, cut_outliers=3):
    if report_file == 'stdout':
        out_file = sys.stdout
    else:
        out_file = open(report_file, 'w')
    max_len = max(map(len, results.keys()))
    string_patter = '{:' + str(max_len) + '} | {:.4f} | {:.4f}\n'
    scores = {}
    all_data_scores = []
    for name, (val_scores, test_scores) in results.items():
        all_data_scores.append(test_scores[:, -1])
        if name == RANDOM_NAME:
            continue

        best_iterations = np.argmax(val_scores, axis=1)
        test_scores_on_best_iter = test_scores[np.arange(len(test_scores)), best_iterations]

        scores[name] = _get_mean_std(test_scores_on_best_iter, cut_outliers)

    scores[ALL_DATA] = _get_mean_std(np.concatenate(all_data_scores), len(all_data_scores) * cut_outliers)

    for name, (mean, std) in sorted(scores.items(), key=lambda x: -x[1][0]):
        out_file.write(string_patter.format(name, mean, std))

    if report_file != 'stdout':
        out_file.close()


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
    single_result_parser.add_argument('--no_draw', dest='draw_plots', action='store_false')
    single_result_parser.add_argument('--print_test_scores', action='store_true')
    single_result_parser.add_argument('--report_file', default='stdout')
    single_result_parser.add_argument('--cut_outliers', type=int, default=3)

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
        results, params = get_results(args.results_path, args.strategies_to_data, args.key_for_name)
        if args.draw_plots:
            plot_data = get_plot_data(
                args.results_path,
                args.key_for_name,
                max_size_draw=args.max_size_draw,
                strategies_to_data=args.strategies_to_data,
                cut_outliers=args.cut_outliers,
                results=results,
                params=params,
            )
            draw_plots(plot_data, args.compare_with_random, args.title, image_path=args.image_path)

        if args.print_test_scores:
            print_test_scores(results, args.report_file, args.cut_outliers)

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
