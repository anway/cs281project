# cs281project
CS281 Final Project: Hyperparameter Optimization of Fractional-Layer Neural Nets

Everything uses the UCI wines data set, which is contained in the folder wines_data.

First we compared the fractional neural net against the regular neural net, where both had a fixed integer part for the number of layers, and the fractional neural net optimized an additional fractional part. These are found in regular_net.py and fractional_net.py. The output can be parsed into graphs using parse_plot_data.py.

Next we performed hyperparameter optimization using the fractional net, and we tried a variety of approaches. We compared against a standard grid search, contained in hyperparam_gridsearch.py. We tried two-loop gradient-based methods, using BFGS for both loops (hyperparam_two_loop_bfgs_bfgs.py), SGD for the inner loop and BFGS for the outer loop (hyperparam_two_loop_sgd_bfgs.py), and we tried optimizing the weights and the layers together in one loop (hyperparam_one_loop.py). To parse the output, we used parse_hyperparam_output.py to parse the result of the two loop runs, and we used parse_largeweights_output.py to parse the results of the one loop run. sample_hyperparam_one_loop_output.txt contains a sample text file for the one loop optimization that we parsed, and sample_hyperparam_two_loop_output.txt contains a sample text file for the two loop optimization that we parsed.
