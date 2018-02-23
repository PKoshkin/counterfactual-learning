from results_utils import read_results, draw_plots, print_stats, MAX_QUERIES_KEY, STRATEGY_KEY


filename = 'al_test_results.txt'
results = read_results(filename)
keys = [key for key in results.keys()
        if key[MAX_QUERIES_KEY] > 40000]
print_stats(results, keys)
draw_plots(
    results,
    keys,
    capsize=3,
    capthick=1.2,
    elinewidth=0.8,
)
