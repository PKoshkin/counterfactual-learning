from results_utils import read_results, draw_plots, print_stats
from results_utils import MAX_QUERIES_KEY, STRATEGY_KEY, INITIAL_SIZE_KEY, BATCH_SIZE_KEY


# filename = 'al_test_results.txt'
filename = 'krkopt_results.txt'
results = read_results(filename)
keys = [key for key in results.keys()
        if key[INITIAL_SIZE_KEY] == 1000]
# print_stats(results, keys)
draw_plots(
    results,
    keys,
    # name_keys=[BATCH_SIZE_KEY],
    thick_num=7,
    title='king-rook vs king',
    ylabel='Accuracy',
    capsize=3,
    capthick=1.2,
    elinewidth=0.8,
)
