import numpy as np
import os
import tensorflow as tf

from replay_buffer import ReplayBuffer
from her import make_sample_her_transitions

from mlp import MLP

DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.8,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 20,  # per epoch
    'rollout_batch_size': 1,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 1024,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.1,  # percentage of time a random action is taken
    'noise_eps': 0.1,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
}

def prepare_params_mlp(kwargs):
    # MLP params
    ddpg_params = dict()

    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params
    return kwargs

def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    tmp_env.reset()
    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def configure_mlp(dims, model_name, model_save_path):
    '''
    def __init__(self, num_states=14, num_actions=9, n_layers=4, hidden_units=[100, 60, 40, 20], act_fn='relu',
                 model_name='il_policy', polyak=0.5, batch_size=20, action_l2=1.0, clip_return=None, clip_pos_returns=None,
                 bc_loss=True, q_filter=False, save_path=MODEL_SAVE_PATH):
    '''
    num_states = dims['o']
    num_actions = dims['u']
    gamma = 1 - 1.0/60

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    policy = MLP(num_states=num_states, num_actions=num_actions, bc_loss=False, gamma=gamma, model_name=model_name,
                 save_path=model_save_path)
    return policy

def configure_ddpg(dims, params, bc_loss, q_filter, num_demo, reuse=False, use_mpi=True, clip_return=True):
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']

    input_dims = dims.copy()

    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = DDPG(reuse=reuse, use_mpi=use_mpi, bc_loss=bc_loss, q_filter=q_filter, num_demo=num_demo, **ddpg_params)
    return policy


ACT_FNS = {'relu': tf.nn.relu}


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

class ILPolicy:
    def __init__(self, num_states=14, num_actions=9, n_layers=4, hidden_units=[100, 60, 40, 20], act_fn='relu', model_name='il_policy',
                 save_path=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.input_dims = {'o' : num_states, 'u' : num_actions, 'r' : 1}
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.act_fn = ACT_FNS.get(act_fn, None)
        self.model_name = model_name
        self.save_path = save_path
        self.sess, self.saver = None, None

        # Creates a comp graph and loads trained model
        self.build()

    def build(self):
        self.x_ = tf.placeholder(tf.float32, [None, self.num_states], 'obs')
        self.y_ = tf.placeholder(tf.float32, [None, self.num_actions], 'ac')
        self.ce_weights_ = tf.placeholder(tf.float32, [None])
        self.layers = []
        h = self.x_
        for i in range(self.n_layers):
            h = tf.layers.dense(h, self.hidden_units[i], activation=self.act_fn)
            self.layers.append(h)
            bn = tf.layers.batch_normalization(h)
            self.layers.append(bn)
        self.y_pred_ = tf.layers.dense(h, self.num_actions)
        self.loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y_pred_))
        self.opt_op_ = tf.train.AdamOptimizer().minimize(self.loss_)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        #self.buffer = []

        input_shapes = dims_to_shapes(self.input_dims)

        # Configure the replay buffer
        self.rollout_batch_size = 1
        self.buffer_size = int(1E6)
        self.T = 60 # the maximum number of commands to apply (roughly 12 sec since it's 5 Hz)

        # The code below is for python3.x
        # buffer_shapes = {key: (self.T if key != 'o' else self.T + 1, *input_shapes[key])
        #                  for key, val in input_shapes.items()}
        buffer_shapes = {}
        for key, val in input_shapes.items():
            if key != 'o':
                temp_T = self.T
            else:
                temp_T = self.T + 1
            expanded_shape = [temp_T] + list(input_shapes[key]) #I'm hoping input_shapes returns a list..-> was tuple
            buffer_shapes[key] = tuple(expanded_shape)

        #buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        #buffer_shapes['ag'] = (self.T + 1, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size

        print("Printing buffer_shapes... {}".format(buffer_shapes))
        self.buffer = ReplayBuffer(buffer_size)

        self.sess.run(tf.global_variables_initializer())

    def load_model(self, model_name=None):
        model_name = model_name or self.model_name
        self.saver.restore(self.sess, os.path.join(self.save_path, model_name))
        print('model restored')

    def save_model(self, model_name=None):
        """returns save path"""
        model_name = model_name or self.model_name
        return self.saver.save(self.sess, os.path.join(self.save_path, model_name))

    def get_actions(self, o, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        return self.eval([o])[0]

    def store_episode(self, episode):
        self.buffer.store_episode(episode)
        #self.buffer.append(episode)
        print("storing episode. episode u is: {}".format(episode['u'].shape))
        print("storing episode. episode o is: {}".format(episode['o'].shape))
        print("storing episode. episode r is: {}".format(episode['r'].shape))

    def step(self, batch_x, batch_y):
        loss, _ = self.sess.run([self.loss_, self.opt_op_], feed_dict={self.x_: batch_x, self.y_: batch_y})
        return loss

    def train(self, dataset, num_steps=20000, batch_size=32):
        # dataset is of DataSet class
        losses, accuracies = [], []
        best_acc = .85
        for step in range(num_steps):
            batch_x, batch_y = dataset.next_batch(batch_size)
            loss = self.step(batch_x, batch_y)
            if (step + 1) % 1000:
                acc = self.score(dataset.test_data, dataset.test_labels)
                accuracies.append(acc)
                if acc > best_acc:
                    best_acc = acc
                    save_loc = self.save_model()
                    print('saved model at {} after {} steps'.format(save_loc, step))
            losses.append(loss)
        return losses, accuracies

    def score(self, x, y):
        y_pred = self.eval(x)

        true_classes = np.argmax(y, axis=1)
        acc = np.sum(true_classes == y_pred) / float(len(y_pred))
        return acc

    def eval(self, obs):
        # takes in obs, returns action
        action = self.sess.run(self.y_pred_, feed_dict={self.x_: obs})
        return np.argmax(action, axis=-1)

def configure_simple_mlp(dims, model_name, model_save_path):
    #rollout_batch_size = params['rollout_batch_size']
    #mlp_params = params['mlp_params']
    #policy = MLP(**mlp_params)
    num_states = dims['o']
    num_actions = dims['u']
    policy = ILPolicy(num_states=num_states, num_actions=num_actions, model_name=model_name,
                      save_path=model_save_path)
    return policy


