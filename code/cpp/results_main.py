from results_utils import read_results, draw_plots, print_stats


filename = 'al_test_results.txt'
results = read_results(filename)
print_stats(results)
draw_plots(results)
