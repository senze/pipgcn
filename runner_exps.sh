###  This small script runs the experiments from our paper.  ###
###  Uncomment a line to run the respective experiment.      ###
###  These experiments are actually run multiple times,      ###
###  once for each set of random seeds in configuration.py.  ###
###  For this reason we recommend not uncommenting more      ###
###  than one line at a time.                                ###

python runner.py no_conv.yml
# python runner.py single_weight_matrix.yml
# python runner.py node_avg.yml
# python runner.py node_edge_avg.yml
# python runner.py order_dependent.yml
# python runner.py deep_tensor.yml
# python runner.py dcnn_hop2.yml
# python runner.py dcnn_hop5.yml
