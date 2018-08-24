import tensorflow as tf
from models import get_network_builder
from utils import fc
from tf_util import adjust_shape

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)


class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def pdfromlatent(self, latent_vector):
        raise NotImplementedError

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return CategoricalPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = fc(latent_vector, 'pi', self.ncat, init_scale=init_scale, init_bias=init_bias)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=one_hot_actions)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, num_actions, observations, latent, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment
        observations    tensorflow placeholder in which the observations will be fed
        latent          latent state from which policy distribution parameters should be inferred
        vf_latent       latent state from which value function should be inferred (if None, then latent is used)
        sess            tensorflow session to run calculations in (if None, default session is used)
        **tensors       tensorflow tensors for additional attributes such as state or mask
        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        self.pdtype = CategoricalPdType(num_actions)
        #self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        self.action = self.pd.sample()
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess

        if estimate_q:
            # assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', num_actions)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:, 0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observaion(s)
        Parameters:
        ----------
        observation     observation data (either single or a batch)
        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)
        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation,
                                              **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observaion(s)
        Parameters:
        ----------
        observation     observation data (either single or a batch)
        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)
        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        pass
        #tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        pass
        #tf_util.load_state(load_path, sess=self.sess)

#def build_policy(env, policy_network, value_network=None, normalize_observations=False, estimate_q=False, **policy_kwargs):
def build_policy(policy_network, value_network=None, normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    # def policy_fn(num_states=9, num_actions=9, sess=None):
    def policy_fn(observ_placeholder, num_actions, sess=None):
        #X = tf.placeholder(shape=(None, num_states), dtype=tf.int32, name='Ob')
        X = observ_placeholder

        extra_tensors = {}

        encoded_x = tf.to_float(X)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent, recurrent_tensors = policy_network(encoded_x) # recurrent_tensors should be None for mlp

        _v_net = value_network # check if value_network is None for acer -> it's None, meaning network params are shared with policy network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                vf_latent, _ = _v_net(encoded_x)

        policy = PolicyWithValue(
            num_actions=num_actions,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn