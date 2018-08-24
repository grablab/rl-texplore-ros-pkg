import tensorflow as tf
import numpy as np
from utils import fc


def mlp(num_layers=2, num_hidden=64, activation=tf.nn.relu):
    """
    Simple fully connected layer policy. Separate stacks of fully-connected layers are used for policy and value function estimation.
    More customized fully-connected policies can be obtained by using PolicyWithV class directly.
    Parameters:
    ----------
    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------
    function that builds fully connected network with a given input placeholder
    """

    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = activation(fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2)))
        return h, None

    return network_fn


def get_network_builder(name):
    # TODO: replace with reflection?
    if name == 'mlp':
        return mlp
    else:
        raise ValueError('Unknown network type: {}'.format(name))
