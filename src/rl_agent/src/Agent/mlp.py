import numpy as np
import tensorflow as tf
from collections import OrderedDict
#from replay_buffer import PrioritizedReplayBuffer
from tensorflow.contrib.staging import StagingArea
from replay_buffer import ReplayBuffer
from util import flatten_grads, import_function

from mpi_adam import MpiAdam

ACT_FNS = {'relu': tf.nn.relu}

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

class MLP(object):
    def __init__(self, num_states=14, num_actions=9, buffer_size=10000, n_layers=4, hidden_units=[100, 60, 40, 20], act_fn='relu',
                 model_name='il_policy', polyak=0.5, batch_size=40, demo_batch_size=30, action_l2=1.0, clip_return=None,
                 clip_pos_returns=None, bc_loss=True, q_filter=False, gamma=None, save_path=None, reuse=False):
        # store args
        self.num_states = num_states; self.num_actions = num_actions; self.n_layers = n_layers
        self.hidden_units = hidden_units; self.model_name = model_name; self.save_path = save_path
        self.batch_size = batch_size; self.demo_batch_size= demo_batch_size; self.clip_return = clip_return;
        self.clip_pos_returns = clip_pos_returns
        self.bc_loss = bc_loss; self.q_filter = q_filter; self.action_l2 = action_l2; self.polyak=polyak
        self.gamma = gamma; self.buffer_size = buffer_size
        self.Q_lr = 0.01; self.pi_lr = 0.01

        self.rollout_batch_size = 1

        if self.clip_return is None:
            self.clip_return = np.inf

        self.act_fn = ACT_FNS.get(act_fn, None)
        self.input_dims = {'o' : num_states, 'u' : num_actions, 'r' : 1}
        self.scope = 'mlp'
        self.network_class = 'actor_critic:ActorCritic'   # I need to check if this works with import function thing.
        self.create_actor_critic = import_function(self.network_class)

        self.sess, self.saver = None, None

        # Do I need this? -> Check what self.network_class is.
        #self.create_actor_critic = import_function(self.network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimr = self.input_dims['r']
        self.dimu = self.input_dims['u']

        self.lambda1 = 0.001
        self.lambda2 = 0.0078

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            print('input_shapes[key]: {}'.format(input_shapes[key]))
            stage_shapes[key] = (None, input_shapes[key][0])
        for key in ['o']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer
        ''' This is for python3.x
        buffer_shapes = {key: (self.T if key != 'o' else self.T + 1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        '''#Below is for python2.x # But it's for a fixed T setting.
        '''
        buffer_shapes = {}
        for key, val in input_shapes.items():
            if key != 'o':
                temp_T = self.T
            else:
                temp_T = self.T + 1
            expanded_shape = [temp_T] + input_shapes[key] #I'm hoping input_shapes returns a list..
            buffer_shapes[key] = tuple(expanded_shape)
        #buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        #buffer_shapes['ag'] = (self.T + 1, self.dimg)
        '''

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size

        self.buffer = ReplayBuffer(buffer_size)
        #TODO: Check if replay_strategy = None is enough to do a regular uniform experience replay
        # self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)
        ## TODO: 7/27 I have to change some of ReplayBuffer in HER to DeepQ type PriortizedReplayBuffer
        # After this, everything should be ready. finally.
        #self.buffer = PrioritizedReplayBuffer(buffer_shapes, )

        # I don't think I need demoBuffer for now -> This is necessary if I want to use self.bc_loss = True

    def _create_network(self, reuse):
        #ToDo: Read logger.py in baselines in OpenAI github repo
        #logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        print("Creating a simple DDPG agent..")
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession() # what is this for?

        # running averages
        # TODO: This is for getting stats. Might skip this for now

        # mini-batch sampling
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1]) # TODO: I have to change this reward value thing?

        # #TODO does this mean that I need to add demo buffer?
        mask = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)), axis=0)

        # networks
        #print("printing self.__dict__: {}".format(self.__dict__))
        with tf.variable_scope('main') as vs:
            if reuse: #TODO: Understand what this reuse thing does.
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, dimo=self.input_dims['o'], dimu=self.input_dims['u'], hidden=10, layers=3)
            vs.reuse_variables()

        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            self.target = self.create_actor_critic(
                target_batch_tf, dimo=self.input_dims['o'], dimu=self.input_dims['u'], hidden=10, layers=3)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, clip_range[0], clip_range[1])  # y = r + gamma*Q(pi)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))  # (y-Q(critic))^2

        # add bc_loss and q_filter stuff here
        if self.bc_loss == 1 and self.q_filter == 1:
            # where is the demonstrator action better than actor action according to the critic?
            maskMain = tf.reshape(tf.boolean_mask(self.main.Q_tf > self.main.Q_pi_tf, mask), [-1])
            self.cloning_loss_tf = tf.reduce_sum(tf.square(
                tf.boolean_mask(tf.boolean_mask((self.main.pi_tf), mask), maskMain, axis=0) - tf.boolean_mask(
                    tf.boolean_mask((batch_tf['u']), mask), maskMain, axis=0)))
            self.pi_loss_tf = -self.lambda1 * tf.reduce_mean(self.main.Q_pi_tf)
            self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf))
            self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf
        elif self.bc_loss == 1 and self.q_filter == 0:
            self.cloning_loss_tf = tf.reduce_sum(
                tf.square(tf.boolean_mask((self.main.pi_tf), mask) - tf.boolean_mask((batch_tf['u']), mask)))
            self.pi_loss_tf = -self.lambda1 * tf.reduce_mean(self.main.Q_pi_tf)
            self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf))
            self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf
        else:
            self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
            self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf))
            self.cloning_loss_tf = tf.reduce_sum(tf.square(self.main.pi_tf - batch_tf['u']))  # random

        # TODO Add grads tf
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # TODO Add optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        # self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]),
                zip(self.target_vars, self.main_vars)))

        # init all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def store_episode(self, episode_batch, update_stats=True):
        self.buffer.store_episode(episode_batch)

    def get_actions(self, o, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        # o = self._preprocess_o(o)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        print("dimension of o in get_actions: {}".format(np.array(o)))
        feed = {
            policy.o_tf: np.array(o).reshape(-1, self.dimo),
            policy.u_tf: np.zeros((1, self.dimu), dtype=np.float32)
        }
        ret = self.sess.run(vals, feed_dict=feed)
        # action post processing
        u = ret[0]
        # add noise for exploratin
        ###

        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episodes(self):
        pass

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        # TODO Fix bug here
        # Just print everything!!
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def sample_batch(self):
        if self.bc_loss:
            #TODO: I can probably just use train.csv to fill up demoBuffer.
            transitions = self.buffer.sample(self.batch_size - self.demo_batch_size)
        else:
            transitions = self.buffer.sample(self.batch_size)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss


    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss , actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.pi_loss_tf,
            self.Q_grad_tf,
            self.pi_grad_tf
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad


    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)