import tensorflow as tf
from util import nn


class ActorCritic:
    def __init__(self, inputs_tf, dimo, dimu, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.
        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        #o = self.o_stats.normalize(self.o_tf)
        #g = self.g_stats.normalize(self.g_tf)
        #input_pi = tf.concat(axis=1, values=[o, g])  # for actor
        input_pi = self.o_tf

        # Networks.
        with tf.variable_scope('pi'):
            # self.pi_tf would be output logits before softmax
            self.pi_tf = nn(input_pi, [self.hidden] * self.layers + [self.dimu])
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, self.pi_tf]) #actions from the policy
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, self.u_tf]) #actions from the buffer
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)