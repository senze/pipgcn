import numpy as np
import tensorflow as tf

# export modules
__all__ = [
    'no_convolution',
    'single_weight_matrix',
    'node_average',
    'node_edge_average',
    'order_dependent',
    'deep_tensor_convolution',
    'diffusion_convolution',
    'merge',
    'dense',
    'average_predictions',
    'initializer',
    'nonlinearity'
]

""" ===== Layers ===== """
""" All layers have as first two parameters:
        - inputs: input tensor or tuple of input tensors
        - params: dictionary of parameters, could be None
    and return tuple containing:
        - outputs: output tensor or tuple of output tensors
        - params: dictionary of parameters, could be None
"""


def no_convolution(inputs, params, filters=None, keep_prob=1.0, trainable=True, **kwargs):
    """ No Convolution: z_c = sigma(W * x_c + b)
    :param inputs: vertices, edges(not used), hood_indices(not used)
    :param params: shared params or None(need to create new params)
    :param filters: filter dimension
    :param keep_prob: dropout_keep_prob
    :param trainable:
    :param kwargs:
    :return:
    """
    vertices = inputs[0]
    vertices = tf.nn.dropout(vertices, keep_prob)
    vertex_shape = vertices.get_shape()
    if params is None:
        # create new weights
        weights = tf.Variable(initializer('uniform', (vertex_shape[1].value, filters)),
                              name='weights', trainable=trainable)  # (vertex_dimension, filters)
        biases = tf.Variable(initializer('zero', (filters,)), name='biases', trainable=trainable)
    else:
        # use shared weights
        weights = params['weights']
        biases = params['biases']
        filters = weights.get_shape()[-1].value
    params = {
        'weights': weights,
        'biases': biases
    }
    # generate vertex signals
    z_centroid = tf.matmul(vertices, weights, name='z_centroid')  # (num_vertices, filters)
    # broadcasting : matrix plus vector
    signals = z_centroid + biases
    sigma = nonlinearity('ReLU')
    z = tf.reshape(sigma(signals), tf.constant([-1, filters]))
    z = tf.nn.dropout(z, keep_prob)
    return z, params


def single_weight_matrix(inputs, params, filters=None, keep_prob=1.0, trainable=True, **kwargs):
    """ Single Weight Matrix: z_c = sigma(W * x_c + avg(W * x_n) + b)
    :param inputs: vertices, edges(not used), hood_indices
    :param params: shared params or None(need to create new params)
    :param filters: filter dimension
    :param keep_prob: dropout_keep_prob
    :param trainable:
    :param kwargs:
    :return:
    """
    vertices = inputs[0]
    vertex_shape = vertices.get_shape()
    hood_indices = inputs[2]
    hood_indices = tf.squeeze(hood_indices, axis=2)
    # for fixed number of neighbors, -1 is a pad value
    hood_sizes = tf.expand_dims(tf.count_nonzero(hood_indices + 1, axis=1, dtype=tf.float32), -1)
    if params is None:
        # create new weights
        weights = tf.Variable(initializer('uniform', (vertex_shape[1].value, filters)),
                              name='weights', trainable=trainable)  # (vertex_dimension, filters)
        biases = tf.Variable(initializer('zero', (filters,)), name='biases', trainable=trainable)
    else:
        # use shared weights
        weights = params['weights']
        biases = params['biases']
        filters = weights.get_shape()[-1].value
    params = {
        'weights': weights,
        'biases': biases
    }
    # generate vertex signals
    z_centroid = tf.matmul(vertices, weights, name='z_centroid')  # (num_vertices, filters)
    # create neighbor signals
    z_vertex = tf.matmul(vertices, weights, name='z_vertex')  # (num_vertices, filters)
    z_neighbor = tf.divide(tf.reduce_sum(tf.gather(z_vertex, hood_indices), 1),
                           tf.maximum(hood_sizes, tf.ones_like(hood_sizes)))  # (num_vertices, vertex_filters)
    # broadcasting : matrix plus vector
    signals = z_centroid + z_neighbor + biases
    sigma = nonlinearity('ReLU')
    z = tf.reshape(sigma(signals), tf.constant([-1, filters]))
    z = tf.nn.dropout(z, keep_prob)
    return z, params


def node_average(inputs, params, filters=None, keep_prob=1.0, trainable=True, **kwargs):
    """ Node Average: z_centroid = sigma(W_c * x_c + avg(W_n * x_n) + b)
    :param inputs: vertices, edges(not used), hood_indices
    :param params: shared params or None(need to create new params)
    :param filters: filter dimension
    :param keep_prob: dropout_keep_prob
    :param trainable:
    :param kwargs:
    :return:
    """
    vertices = inputs[0]
    vertex_shape = vertices.get_shape()
    hood_indices = inputs[2]
    hood_indices = tf.squeeze(hood_indices, axis=2)
    # for fixed number of neighbors, -1 is a pad value
    hood_sizes = tf.expand_dims(tf.count_nonzero(hood_indices + 1, axis=1, dtype=tf.float32), -1)
    if params is None:
        # create new weights
        weights_c = tf.Variable(initializer('uniform', (vertex_shape[1].value, filters)),
                                name='weights_c', trainable=trainable)  # (vertex_dimension, filters)
        weights_n = tf.Variable(initializer('uniform', (vertex_shape[1].value, filters)),
                                name='weights_n', trainable=trainable)  # (vertex_dimension, filters)
        biases = tf.Variable(initializer('zero', (filters,)), name='biases', trainable=trainable)
    else:
        weights_c = params['weights_c']
        weights_n = params['weights_n']
        biases = params['biases']
        filters = weights_c.get_shape()[-1].value
    params = {
        'weights_c': weights_c,
        'weights_n': weights_n,
        'biases': biases
        }
    # generate vertex signals
    z_centroid = tf.matmul(vertices, weights_c, name='z_centroid')  # (num_vertices, filters)
    # create neighbor signals
    z_vertex = tf.matmul(vertices, weights_n, name='z_vertex')  # (num_vertices, filters)
    z_neighbor = tf.divide(tf.reduce_sum(tf.gather(z_vertex, hood_indices), 1),
                           tf.maximum(hood_sizes, tf.ones_like(hood_sizes)))  # (num_vertices, vertex_filters)
    # broadcasting : matrix plus vector
    signals = z_centroid + z_neighbor + biases
    sigma = nonlinearity('ReLU')
    z = tf.reshape(sigma(signals), tf.constant([-1, filters]))
    z = tf.nn.dropout(z, keep_prob)
    return z, params


def node_edge_average(inputs, params, filters=None, keep_prob=1.0, trainable=True, **kwargs):
    """ Node Edge Average: z_c = sigma(W_c * x_c + avg(W_n * x_n) + avg(W_e * a_n) + b)
    :param inputs: vertices, edges, hood_indices
    :param params: shared params or None(need to create new params)
    :param filters: filter dimension
    :param keep_prob: dropout_keep_prob
    :param trainable:
    :param kwargs:
    :return:
    """
    vertices = inputs[0]
    vertex_shape = vertices.get_shape()
    edges = inputs[1]
    edge_shape = edges.get_shape()
    hood_indices = inputs[2]
    hood_indices = tf.squeeze(hood_indices, axis=2)
    # for fixed number of neighbors, -1 is a pad value
    hood_sizes = tf.expand_dims(tf.count_nonzero(hood_indices + 1, axis=1, dtype=tf.float32), -1)
    if params is None:
        # create new weights
        weights_c = tf.Variable(initializer('uniform', (vertex_shape[1].value, filters)),
                                name='weights_c', trainable=trainable)  # (vertex_dimension, filters)
        weights_n = tf.Variable(initializer('uniform', (vertex_shape[1].value, filters)),
                                name='weights_n', trainable=trainable)  # (vertex_dimension, filters)
        weights_e = tf.Variable(initializer('uniform', (edge_shape[2].value, filters)),
                                name='weights_e', trainable=trainable)  # (edge_dimension, filters)
        biases = tf.Variable(initializer('zero', (filters,)), name='biases', trainable=trainable)
    else:
        # use shared weights
        weights_c = params['weights_c']
        weights_n = params['weights_n']
        weights_e = params['weights_e']
        biases = params['biases']
        filters = weights_c.get_shape()[-1].value
    params = {
        'weights_c': weights_c,
        'weights_n': weights_n,
        'weights_e': weights_e,
        'biases': biases
        }
    # generate vertex signals
    z_centroid = tf.matmul(vertices, weights_c, name='z_centroid')  # (num_vertices, filters)
    # create neighbor signals
    z_vertex = tf.matmul(vertices, weights_n, name='z_vertex')  # (num_vertices, filters)
    z_edge = tf.tensordot(edges, weights_e, axes=[[2], [0]], name='z_edge')  # (num_vertices, num_neighbors, filters)
    z_neighbor = tf.divide(tf.reduce_sum(tf.gather(z_vertex, hood_indices), 1) + tf.reduce_sum(z_edge, 1),
                           tf.maximum(hood_sizes, tf.ones_like(hood_sizes)))  # (num_vertices, vertex_filters)
    # broadcasting : matrix plus vector
    signals = z_centroid + z_neighbor + biases
    sigma = nonlinearity('ReLU')
    z = tf.reshape(sigma(signals), tf.constant([-1, filters]))
    z = tf.nn.dropout(z, keep_prob)
    return z, params


def order_dependent(inputs, params, filters=None, keep_prob=1.0, trainable=True, **kwargs):
    """ Order Dependent: z_c = sigma(W_c * x_c + avg(W_nj * x_nj) + avg(W_ej * a_nj) + b)
    :param inputs: vertices, edges, hood_indices
    :param params: shared params or None(need to create new params)
    :param filters: filter dimension
    :param keep_prob: dropout_keep_prob
    :param trainable:
    :param kwargs:
    :return:
    """
    vertices = inputs[0]
    vertex_shape = vertices.get_shape()
    edges = inputs[1]
    edge_shape = edges.get_shape()
    hood_indices = inputs[2]
    hood_indices = tf.squeeze(hood_indices, axis=2)
    hood_size = hood_indices.get_shape()[1].value
    if params is None:
        # create new weights
        weights_c = tf.Variable(initializer('uniform', (vertex_shape[1].value, filters)),
                                name='weights_c', trainable=trainable)  # (vertex_dimension, filters)
        weights_n = [tf.Variable(initializer('uniform', (vertex_shape[1].value, filters)),
                                 name='weights_n{}'.format(i),
                                 trainable=trainable) for i in range(hood_size)]  # (vertex_dimension, filters)
        weights_e = tf.Variable(initializer('uniform', (hood_size, edge_shape[2].value, filters)),
                                name='weights_e', trainable=trainable)  # (num_neighbors, edge_dimension, filters)
        biases = tf.Variable(initializer('zero', (filters,)), name='biases', trainable=trainable)
    else:
        # use shared weights
        weights_c = params['weights_c']
        weights_n = [params['weights_n{}'.format(i)] for i in range(hood_size)]
        weights_e = params['weights_e']
        biases = params['biases']
        filters = weights_c.get_shape()[-1].value
    params = {
        'weights_c': weights_c,
        'weights_e': weights_e,
        'biases': biases
    }
    params.update({'weights_n{}'.format(i): weights_n[i] for i in range(hood_size)})
    # generate vertex signals
    z_centroid = tf.matmul(vertices, weights_c, name='z_centroid')  # (num_vertices, filters)
    # create neighbor signals
    # for each neighbor, calculate signals:
    z_vertex = tf.zeros_like(z_centroid)
    for i in range(hood_size):
        z_vertex += tf.matmul(tf.gather(vertices, hood_indices[:, i]), weights_n[i])
    z_vertex = tf.divide(z_vertex, tf.constant(hood_size, dtype=tf.float32))
    z_edge = tf.tensordot(edges, weights_e, axes=[[1, 2], [0, 1]])  # (num_vertices, filters)
    z_edge = tf.divide(z_edge, tf.constant(hood_size, dtype=tf.float32))
    # broadcasting : matrix plus vector
    signals = z_centroid + z_vertex + z_edge + biases
    sigma = nonlinearity('ReLU')
    z = tf.reshape(sigma(signals), tf.constant([-1, filters]))
    z = tf.nn.dropout(z, keep_prob)
    return z, params


def deep_tensor_convolution(inputs, params, factors=None, keep_prob=1.0, trainable=True, **kwargs):
    """ Deep Tensor Convolution: z_c = x_c + avg(sigma(W_f * (W_n * x_n + b_n)⊕(W_e * a_n + b_e))
    :param inputs: vertices, edges, hood_indices
    :param params: shared params or None(need to create new params)
    :param factors:
    :param keep_prob: dropout_keep_prob
    :param trainable:
    :param kwargs:
    :return:
    """
    vertices = inputs[0]
    vertex_shape = vertices.get_shape()
    edges = inputs[1]
    edge_shape = edges.get_shape()
    hood_indices = inputs[2]
    hood_indices = tf.squeeze(hood_indices, axis=2)
    hood_size = hood_indices.get_shape()[1].value
    if params is None:
        # create new weights
        weights_f = tf.Variable(initializer('uniform', (factors, vertex_shape[1].value)),
                                name='weights_f', trainable=trainable)  # (factors, vertex_dimension)
        weights_n = tf.Variable(initializer('uniform', (vertex_shape[1].value, factors)),
                                name='weights_n', trainable=trainable)  # (vertex_dimension, factors)
        weights_e = tf.Variable(initializer('uniform', (edge_shape[2].value, factors)),
                                name='weights_e', trainable=trainable)  # (edge_dimension, factors)
        biases_n = tf.Variable(initializer('zero', (factors,)), name='biases_n', trainable=trainable)
        biases_e = tf.Variable(initializer('zero', (factors,)), name='biases_e', trainable=trainable)
    else:
        # use shared weights
        weights_f = params['weights_f']
        weights_n = params['weights_n']
        weights_e = params['weights_e']
        biases_n = params['biases_n']
        biases_e = params['biases_e']
    params = {
        'weights_f': weights_f,
        'weights_n': weights_n,
        'weights_e': weights_e,
        'biases_n': biases_n,
        'biases_e': biases_e
    }
    # create neighbor signals
    z_vertex = tf.matmul(vertices, weights_n, name='z_vertex')  # (num_vertices, factors)
    z_vertex += biases_n
    z_edge = tf.tensordot(edges, weights_e, axes=[[2], [0]], name='z_edge')  # (num_vertices, num_neighbors, factors)
    z_edge += biases_e
    z_factor = tf.gather(z_vertex, hood_indices) * z_edge  # (num_vertices, num_neighbors, factors)
    sigma = nonlinearity('tanh')
    z_ij = sigma(tf.tensordot(z_factor, weights_f, axes=[[2], [0]],
                              name='z_ij'))  # (num_vertices, num_neighbors, vertex_dimension)
    # broadcasting : matrix plus vector
    signals = vertices + tf.divide(tf.reduce_sum(z_ij, axis=1),
                                   tf.constant(hood_size, dtype=tf.float32))  # (num_vertices, vertex_dimension)
    z = tf.reshape(signals, tf.constant([-1, vertex_shape[1].value]))
    z = tf.nn.dropout(z, keep_prob)
    return z, params


def diffusion_convolution(inputs, params, keep_prob=1.0, trainable=True, **kwargs):
    """ Diffusion Convolution:
    :param inputs: vertices, edges, hood_indices
    :param params: shared params or None(need to create new params)
    :param keep_prob: dropout_keep_prob
    :param trainable:
    :param kwargs:
    :return:
    """
    vertices = inputs[0]
    inputs_dimension = vertices.get_shape()[-1].value
    hops = inputs[1]
    hop = hops.get_shape()[1].value
    if params is None:
        # create new weights
        weights = tf.Variable(initializer('uniform', (1, 1, hop, inputs_dimension)),
                              name='weights', trainable=trainable)
    else:
        # use shared weights
        weights = params['weights']
    params = {
        'weights': weights
    }
    PX = tf.expand_dims(tf.tensordot(hops, vertices, axes=[[-1], [0]]), axis=1)
    z = weights * PX
    z = tf.reshape(z, shape=[-1, inputs_dimension * hop])
    z = tf.nn.dropout(z, keep_prob)
    sigma = nonlinearity('ReLU')
    return sigma(z), params


def merge(inputs, params, **kwargs):
    """ Merge Layer(2 legs: ligand protein and receptor protein)
    :param inputs: input1, input2, examples
    :param params: shared params(not used)
    :param kwargs:
    :return:
    """
    input1, input2, examples = inputs
    out1 = tf.gather(input1, examples[:, 0])
    out2 = tf.gather(input2, examples[:, 1])
    output1 = tf.concat([out1, out2], axis=0)
    output2 = tf.concat([out2, out1], axis=0)
    return tf.concat((output1, output2), axis=1), None


def dense(inputs, params, outputs_dimension=None, keep_prob=1.0, active=True, trainable=True, **kwargs):
    """ Dense Layer(Part of Full-connected Layer)
    :param inputs:
    :param params:
    :param outputs_dimension:
    :param keep_prob:
    :param active:
    :param trainable:
    :param kwargs:
    :return:
    """
    inputs = tf.nn.dropout(inputs, keep_prob)
    inputs_dimension = inputs.get_shape()[-1].value
    outputs_dimension = inputs_dimension if outputs_dimension is None else outputs_dimension
    if params is None:
        weights = tf.Variable(initializer('uniform', [inputs_dimension, outputs_dimension]),
                              name='weights', trainable=trainable)
        biases = tf.Variable(initializer('zero', [outputs_dimension]), name='biases', trainable=trainable)
        params = {'weights': weights, 'biases': biases}
    else:
        weights, biases = params['weights'], params['biases']
    z = tf.matmul(inputs, weights) + biases
    if active:
        active_function = nonlinearity('ReLU')
        z = active_function(z)
    z = tf.nn.dropout(z, keep_prob)
    return z, params


""" ===== Non Layers ===== """


def average_predictions(inputs, params, **kwargs):
    combined = tf.reduce_mean(tf.stack(tf.split(inputs, 2)), 0)
    return combined, None


def initializer(opt, shape):
    """ Initialize 'zero' or uniform shape 'tensorFlow' object
    :param opt: option
    :param shape: shape params
    :return:
    """
    if opt == 'zero':
        return tf.zeros(shape)
    elif opt == 'uniform':
        F_in = np.prod(shape[0:-1])
        std = 1/np.sqrt(F_in)
        return tf.random_uniform(shape, minval=-std, maxval=std)


def nonlinearity(opt):
    """ add non-linearity
    :param opt: option
    :return:
    """
    if opt == 'ReLU':
        return tf.nn.relu
    elif opt == 'tanh':
        return tf.nn.tanh
    elif opt == 'linear' or 'none':
        return lambda x: x
