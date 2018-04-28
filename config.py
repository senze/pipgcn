import os
import datetime
import numpy as np
import tensorflow as tf

"""
Basic configurations and functions
"""

# ensure any existing 'tensorFlow sessions' are closed
try:
    tf.Session().close()
except ConnectionError:
    pass
# directories(operation system environment)
os.environ["PL_EXPERIMENTS"] = "/home/senze/PycharmProjects/pipGCN/experiments/"  # contains experiment yml files
exp_dir = os.getenv("PL_EXPERIMENTS")
os.environ["PL_DATA"] = "/home/senze/PycharmProjects/pipGCN/data/"  # contains pickle data files
data_dir = os.getenv("PL_DATA")
os.environ["PL_OUT"] = "/home/senze/PycharmProjects/pipGCN/out/"  # contains experiment results
out_dir = os.getenv("PL_OUT")
# set 'numpy' options
np.set_printoptions(precision=3)
# random seeds of 'tensorFlow' and 'numpy'
# each random seed represents an experimental replication
# you can add or remove list elements to change the number of replications for an experiment
seeds = [
    {"tf_seed": 649737, "np_seed": 29820},
    # {"tf_seed": 395408, "np_seed": 185228},
    # {"tf_seed": 252356, "np_seed": 703889},
    # {"tf_seed": 343053, "np_seed": 999360},
    # {"tf_seed": 743746, "np_seed": 67440},
    # {"tf_seed": 175343, "np_seed": 378945},
    # {"tf_seed": 856516, "np_seed": 597688},
    # {"tf_seed": 474313, "np_seed": 349903},
    # {"tf_seed": 838382, "np_seed": 897904},
    # {"tf_seed": 202003, "np_seed": 656146},
]


# a slightly fancy printing method
def info(msg):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("{}| {}".format(time_str, msg))
