import os
import sys
import yaml
import traceback
import pickle
import tensorflow as tf
import numpy as np
from config import exp_dir, data_dir, out_dir, seeds, info
from fitter import Fitter
from classifier import Classifier
from processor import Processor

"""
Runner of experiments
Including:
    Building of models  --classifier.py
    Training and testing of models  --fitter.py
    Process and output of results  --processor.py
"""

# load experiment specified in system args
# exp_file = sys.argv[1]
# exp_file = "no_conv.yml"
exp_file = "single_weight_matrix.yml"
# exp_file = "node_avg.yml"
# exp_file = "node_edge_avg.yml"
# exp_file = "order_dependent.yml"
# exp_file = "deep_tensor.yml"
# exp_file = "dcnn_hop2.yml"
# exp_file = "dcnn_hop5.yml"
info("Running Experiment File: {}".format(exp_file))
f_name = exp_file.split('.')[0] if '.' in exp_file else exp_file
exp_spec = yaml.load(open(os.path.join(exp_dir, exp_file), 'r').read())
# ensure output path of results is exist
res_dir = os.path.join(out_dir, f_name)
if not os.path.exists(res_dir):
    os.mkdir(res_dir)
processor = Processor()
# create results log
res_log = os.path.join(res_dir, "results.csv")
with open(res_log, 'w') as f:
    f.write("")
# write experiment specifications to file
with open(os.path.join(res_dir, "exp.yml"), 'w') as f:
    f.write("{}\n".format(yaml.dump(exp_spec)))
# perform each experiment
prev_train_file = None
prev_test_file = None
# create data dictionary
data = {'train': None, 'test': None}
first_exp = True
for name, spec in exp_spec['experiments']:
    train_file = os.path.join(data_dir, spec['train_data_file'])
    test_file = os.path.join(data_dir, spec['test_data_file'])
    try:
        # reuse train data if possible
        if train_file != prev_train_file:
            info("Loading Train Data File")
            data['train'] = pickle.load(open(train_file, 'rb'), encoding='iso-8859-1')[1]
            prev_train_file = train_file
        if test_file != prev_test_file:
            info("Loading Test Data File")
            data['test'] = pickle.load(open(test_file, 'rb'), encoding='iso-8859-1')[1]
            prev_test_file = test_file
        # perform experiment for each random seed
        for i, pair in enumerate(seeds):
            info("{}: rep{}".format(name, i))
            # Set 'tensorFlow' and 'numpy' seeds
            tf.set_random_seed(pair['tf_seed'])
            np.random.seed(int(pair['np_seed']))
            info("Building Model")
            model = Classifier(spec['layers'], spec['layer_args'], data['train'], 0.1, 0.1, res_dir)
            info("Fitting Model")
            headers, results = Fitter(processor).fit(exp_spec, data, model)
            # write headers to file during first experiment
            if first_exp:
                with open(res_log, 'a') as f:
                    f.write("{}\n".format(','.join(['file', 'exp', 'rep', 'spec'] + headers)))
                first_exp = False
            # write results to file
            with open(res_log, 'a') as f:
                f.write("{}, {}, {}, {}, {}\n".format(f_name, name, i, format(spec).replace(',', ';'),
                                                      ','.join([str(r) for r in results])))
    except Exception as er:
        if er is KeyboardInterrupt:
            raise er
        ex_str = traceback.format_exc()
        info(ex_str)
        info("Experiment Failed: {}".format(exp_spec))
