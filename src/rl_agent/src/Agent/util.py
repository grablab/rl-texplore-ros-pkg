import os
import subprocess
import sys
import importlib
import inspect
import functools
import tf_util as U

import tensorflow as tf
import numpy as np

def import_function(spec):
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn

def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def flatten_grads(var_list, grads):
    """Flattens a variables and their gradients.
    """
    return tf.concat([tf.reshape(grad, [U.numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)


def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch

def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input
