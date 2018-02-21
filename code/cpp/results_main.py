from results_utils import read_results, mannwhitneyu_test, draw_plots, print_stats


filename = 'al_test_results.txt'
results = read_results(filename)
print_stats(results)
for _key in results:
    key = _key
mannwhitneyu_test(results, key, key)
draw_plots(results)
