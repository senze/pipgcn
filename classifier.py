import copy
import numpy as np
import tensorflow as tf
import components
from config import info

# export modules
__all__ = ['Classifier']


class Classifier(object):
    """ Builder of models """
    def __init__(self, layers, layer_args, train_data, learning_rate, pn_ratio, res_dir):
        """ Assumes same dimensions and neighborhoods for l_ and r_
        :param layers: exp_spec['spec']->spec['layers']
        :param layer_args: exp_spec['spec']->spec['layer_args']
        :param train_data: exp_spec['train_data_file']->data['train']
        :param learning_rate: Learning Rate
        :param pn_ratio: Positive-Negative Ratio
        :param res_dir: output directory of results
        data: 'l_'->ligand protein; 'r_'->receptor protein
        """
        # parameters & args
        self.layer_args = layer_args
        self.params = {}
        # 'tensorFlow' stuff
        self.graph = tf.Graph()
        self.session = None
        self.predictions = None
        # get details of train data
        self.dcnn = ('l_edge' not in train_data[0])  # convolution type is dcnn or not
        self.vertex_dimension = train_data[0]['l_vertex'].shape[-1]
        if self.dcnn:
            self.hop = train_data[0]['l_power_series'].shape[1]
        else:
            self.edge_dimension = train_data[0]['l_edge'].shape[-1]
            self.hood_size = train_data[0]['l_hood_indices'].shape[1]
        # set self.graph as default Graph Object
        with self.graph.as_default():
            # shapes and 'tensorFlow' variables
            self.vertex1 = tf.placeholder(tf.float32, [None, self.vertex_dimension], 'vertex1')
            self.vertex2 = tf.placeholder(tf.float32, [None, self.vertex_dimension], 'vertex2')
            if self.dcnn:
                self.hop1 = tf.placeholder(tf.float32, [None, self.hop, None], 'hop1')
                self.hop2 = tf.placeholder(tf.float32, [None, self.hop, None], 'hop2')
                input1 = self.vertex1, self.hop1
                input2 = self.vertex2, self.hop2
            else:
                self.edge1 = tf.placeholder(tf.float32, [None, self.hood_size, self.edge_dimension], 'edge1')
                self.edge2 = tf.placeholder(tf.float32, [None, self.hood_size, self.edge_dimension], 'edge2')
                self.hood_indices1 = tf.placeholder(tf.int32, [None, self.hood_size, 1], 'hood_indices1')
                self.hood_indices2 = tf.placeholder(tf.int32, [None, self.hood_size, 1], 'hood_indices2')
                input1 = self.vertex1, self.edge1, self.hood_indices1
                input2 = self.vertex2, self.edge2, self.hood_indices2
            self.examples = tf.placeholder(tf.int32, [None, 2], 'examples')
            self.labels = tf.placeholder(tf.float32, [None], 'labels')
            self.keep_prob = tf.placeholder(tf.float32, [], 'keep_prob')
            # make layers
            legs = True
            inputs = None
            i = 0
            while i < len(layers):
                layer = layers[i]
                args = copy.deepcopy(layer_args)
                args['keep_prob'] = self.keep_prob
                layer_type = layer[0]
                next_arg = layer[1] if len(layer) > 1 else {}
                flag = layer[2] if len(layer) > 2 else None
                args.update(next_arg)
                # getattr(object, name[, default])-> return 'attr' of 'object'
                layer_function = getattr(components, layer_type)  # return layer function
                # flag == ['merge'] if flag is not None
                if flag is not None and 'merge' in flag:
                    legs = False
                    # take vertex features only
                    inputs = input1[0], input2[0], self.examples
                if legs:
                    # make leg layers(everything up to the merge layer)
                    name = 'leg1_{}_{}'.format(layer_type, i)
                    with tf.name_scope(name):
                        output, params = layer_function(input1, None, **args)
                        # update params
                        if params is not None:
                            info("not merge")
                            print(len(params.items()))
                            self.params.update({'{}_{}'.format(name, key): value for key, value in params.items()})
                        if self.dcnn:
                            input1 = output, self.hop1
                        else:
                            input1 = output, self.edge1, self.hood_indices1
                    name = 'leg2_{}_{}'.format(layer_type, i)
                    with tf.name_scope(name):
                        output = layer_function(input2, params, **args)[0]
                        if self.dcnn:
                            input2 = output, self.hop2
                        else:
                            input2 = output, self.edge2, self.hood_indices2
                else:
                    # merged layers
                    name = '{}_{}'.format(layer_type, i)
                    with tf.name_scope(name):
                        inputs, params = layer_function(inputs, None, **args)
                        # update params
                        if params is not None and len(params.items()) > 0:
                            self.params.update({'{}_{}'.format(name, key): value for key, value in params.items()})
                i += 1
            self.predictions = inputs
            # loss
            with tf.name_scope('loss'):
                scale_vector = (pn_ratio * (self.labels - 1) / -2) + ((self.labels + 1) / 2)
                logits = tf.concat([-self.predictions, self.predictions], axis=1)
                labels = tf.stack([(self.labels - 1) / -2, (self.labels + 1) / 2], axis=1)
                self.loss = tf.losses.softmax_cross_entropy(labels, logits, weights=scale_vector)
            # optimizer
            with tf.name_scope('optimizer'):
                # generate an op which trains the model
                self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            # set up 'tensorFlow' session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            # record compute graph:
            # self.summaries = tf.summary.merge_all()
            # self.summary_writer = tf.summary.FileWriter(res_dir, self.graph)

    # def loss(self, data):
    #     return self.run_graph(self.loss, data, 'test')

    """ ===== train ===== """
    def train(self, data):
        return self.run_graph([self.train_op, self.loss], data, 'train')

    """ ===== run graph ===== """
    def run_graph(self, outputs, data, type, options=None, run_metadata=None):
        with self.graph.as_default():
            keep_prob = 1.0
            if type == 'train' and 'keep_prob' in self.layer_args:
                keep_prob = self.layer_args['keep_prob']
            if self.dcnn:
                feed_dict = {
                    self.vertex1: data['l_vertex'],
                    self.vertex2: data['r_vertex'],
                    self.hop1: data['l_power_series'],
                    self.hop2: data['r_power_series'],
                    self.examples: data['label'][:, :2],
                    self.labels: data['label'][:, 2],
                    self.keep_prob: keep_prob}
            else:
                feed_dict = {
                    self.vertex1: data['l_vertex'], self.edge1: data['l_edge'],
                    self.vertex2: data['r_vertex'], self.edge2: data['r_edge'],
                    self.hood_indices1: data['l_hood_indices'],
                    self.hood_indices2: data['r_hood_indices'],
                    self.examples: data['label'][:, :2],
                    self.labels: data['label'][:, 2],
                    self.keep_prob: keep_prob
                }
            return self.sess.run(outputs, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

    # def get_nodes(self):
    #     return [n for n in self.graph.as_graph_def().node]

    """ ===== close ===== """
    def close(self):
        with self.graph.as_default():
            self.sess.close()

    def get_labels(self, data):
        return {'label': data['label'][:, 2, np.newaxis]}

    def predict(self, data):
        results = self.run_graph([self.loss, self.predictions], data, 'test')
        results = {'label': results[1], 'loss': results[0]}
        return results
