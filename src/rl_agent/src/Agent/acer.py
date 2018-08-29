import tensorflow as tf
import numpy as np
from utils import cat_entropy, mse, find_trainable_variables, Scheduler
from utils import get_by_index, q_explained_variance
from utils import cat_entropy_softmax, check_shape, batch_to_seq, seq_to_batch

from her import make_sample_her_transitions
from replay_buffer_her import ReplayBuffer

# remove last step
def strip(var, nenvs, nsteps, flat = False):
    vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
    return seq_to_batch(vars[:-1], flat)

def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
    """
    Calculates q_retrace targets
    :param R: Rewards
    :param D: Dones
    :param q_i: Q values for actions taken
    :param v: V values
    :param rho_i: Importance weight for each action
    :return: Q_retrace values
    """
    rho_bar = batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    rs = batch_to_seq(R, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    ds = batch_to_seq(D, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    q_is = batch_to_seq(q_i, nenvs, nsteps, True)
    vs = batch_to_seq(v, nenvs, nsteps + 1, True)
    v_final = vs[-1]
    qret = v_final
    qrets = []
    for i in range(nsteps - 1, -1, -1):
        check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
        qret = rs[i] + gamma * qret * (1.0 - ds[i])
        qrets.append(qret)
        qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
    qrets = qrets[::-1]
    qret = seq_to_batch(qrets, flat=True)
    return qret

class Model(object):
    def __init__(self, policy, num_states, num_actions, nenvs, nsteps,
                 ent_coef, q_coef, gamma, max_grad_norm, lr,
                 rprop_alpha, rprop_epsilon, total_timesteps, lrschedule,
                 c, trust_region, alpha, delta,
                 buffer_size, bc_loss):
        self.bc_loss = bc_loss
        self.sess = sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession() # what is this for?
        self.nbatch = nbatch = nenvs*nsteps
        #nact = num_actions

        A = tf.placeholder(tf.int32, [nbatch]) # actions
        D = tf.placeholder(tf.float32, [nbatch]) # dones
        R = tf.placeholder(tf.float32, [nbatch]) # rewards, not returns
        MU = tf.placeholder(tf.float32, [nbatch, num_actions]) # mu's
        LR = tf.placeholder(tf.float32, [])
        eps = 1e-6

        # this is going to be X in build_policy
        #step_ob_placeholder = tf.placeholder(dtype=ob_space.dtype, shape=(nenvs,) + ob_space.shape[:-1] + (ob_space.shape[-1] * nstack,))
        #train_ob_placeholder = tf.placeholder(dtype=ob_space.dtype, shape=(nenvs*(nsteps+1),) + ob_space.shape[:-1] + (ob_space.shape[-1] * nstack,))
        step_ob_placeholder = tf.placeholder(dtype=tf.int32, shape=(nenvs,13))
        train_ob_placeholder = tf.placeholder(dtype=tf.int32, shape=(nenvs*(nsteps+1), 13))
        with tf.variable_scope('acer_model', reuse=tf.AUTO_REUSE):
            # step_model = policy(num_states=num_states, num_actions=num_actions, sess=self.sess)
            # step_model = policy(num_states=num_states, num_actions=num_actions, sess=self.sess)
            step_model = policy(observ_placeholder=step_ob_placeholder, num_actions=num_actions, sess=self.sess)
            #train_model = policy(observ_placeholder=train_ob_placeholder, sess=self.sess)
            # train_model = policy(num_states=num_states, num_actions=num_actions, sess=self.sess)
            train_model = policy(observ_placeholder=train_ob_placeholder, num_actions=num_actions, sess=self.sess)

        params = find_trainable_variables("acer_model")
        print("Params {}".format(len(params)))
        for var in params:
            print(var)

        # create polyak averaged model
        ema = tf.train.ExponentialMovingAverage(alpha)
        ema_apply_op = ema.apply(params)

        def custom_getter(getter, *args, **kwargs):
            v = ema.average(getter(*args, **kwargs))
            print(v.name)
            return v

        with tf.variable_scope("acer_model", custom_getter=custom_getter, reuse=True):
            polyak_model = policy(observ_placeholder=train_ob_placeholder, num_actions=num_actions, sess=sess)

        # Notation: (var) = batch variable, (var)s = seqeuence variable, (var)_i = variable index by action at step i

        # action probability distributions according to train_model, polyak_model and step_model
        # poilcy.pi is probability distribution parameters; to obtain distribution that sums to 1 need to take softmax
        train_model_p = tf.nn.softmax(train_model.pi)
        polyak_model_p = tf.nn.softmax(polyak_model.pi)
        step_model_p = tf.nn.softmax(step_model.pi)
        v = tf.reduce_sum(train_model_p * train_model.q, axis = -1) # shape is [nenvs * (nsteps + 1)]

        print("train_model_p shape: {}".format(train_model_p.get_shape().as_list()))
        print("v shape: {}".format(v.get_shape().as_list()))
        print("polyak_model_p shape: {}".format(polyak_model_p.get_shape().as_list()))
        print("train_model.q shape: {}".format(train_model.q.get_shape().as_list()))

        # strip off last step # I'm assuming that the reason you need nsteps+1 for train_model is to get obs_{t+1} info
        # for Experience Replay (each tuple in the buffer should be (o_t, o_{t+1}, r_t, a_t)
        f, f_pol, q = map(lambda var: strip(var, nenvs, nsteps), [train_model_p, polyak_model_p, train_model.q])

        print("f shape: {}".format(f.get_shape().as_list()))
        print("f_pol shape: {}".format(f_pol.get_shape().as_list()))
        print("q shape: {}".format(q.get_shape().as_list()))

        # Get pi and q values for actions taken
        f_i = get_by_index(f, A)  # I might have to take argmax(A) before feeding into this func since it expects scalar
        q_i = get_by_index(q, A)

        # Compute ratios for importance truncation
        rho = f / (MU + eps)
        rho_i = get_by_index(rho, A)

        # Calculate Q_retrace targets
        qret = q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma)

        # Calculate losses
        entropy = tf.reduce_mean(cat_entropy_softmax(f))

        # Policy Graident loss, with truncated importance sampling & bias correction
        v = strip(v, nenvs, nsteps, True)
        check_shape([qret, v, rho_i, f_i], [[nenvs * nsteps]] * 4)
        check_shape([rho, f, q], [[nenvs * nsteps, num_actions]] * 2)

        # Truncated importance sampling
        adv = qret - v
        logf = tf.log(f_i + eps)
        gain_f = logf * tf.stop_gradient(adv * tf.minimum(c, rho_i))  # [nenvs * nsteps]
        loss_f = -tf.reduce_mean(gain_f)


        # Bias correction for the truncation
        adv_bc = (q - tf.reshape(v, [nenvs * nsteps, 1]))  # [nenvs * nsteps, nact]
        logf_bc = tf.log(f + eps) # / (f_old + eps)
        check_shape([adv_bc, logf_bc], [[nenvs * nsteps, num_actions]]*2)
        gain_bc = tf.reduce_sum(logf_bc * tf.stop_gradient(adv_bc * tf.nn.relu(1.0 - (c / (rho + eps))) * f), axis = 1) #IMP: This is sum, as expectation wrt f
        loss_bc = -tf.reduce_mean(gain_bc)

        loss_policy = loss_f + loss_bc

        # Value/Q function loss, and explained variance
        check_shape([qret, q_i], [[nenvs * nsteps]]*2)
        ev = q_explained_variance(tf.reshape(q_i, [nenvs, nsteps]), tf.reshape(qret, [nenvs, nsteps]))
        loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i)*0.5)

        # Net loss
        check_shape([loss_policy, loss_q, entropy], [[]] * 3)
        loss = loss_policy + q_coef * loss_q - ent_coef * entropy

        if trust_region:
            #TODO: Implement trust_region method
            grads = tf.gradients(loss, params)
        else:
            grads = tf.gradients(loss, params)

        if max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=rprop_alpha, epsilon=rprop_epsilon)
        _opt_op = trainer.apply_gradients(grads)

        # so when you call _train, you first do the gradient step, then you apply ema
        with tf.control_dependencies([_opt_op]):
            _train = tf.group(ema_apply_op)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        # Ops/Summaries to run, and their names for logging
        run_ops = [_train, loss, loss_q, entropy, loss_policy, loss_f, loss_bc, ev, norm_grads]
        names_ops = ['loss', 'loss_q', 'entropy', 'loss_policy', 'loss_f', 'loss_bc', 'explained_variance', 'norm_grads']

        if trust_region:
            run_ops = run_ops + []
            names_ops = names_ops + []

        def train(obs, actions, rewards, dones, mus, steps):
            cur_lr = lr.value_steps(steps)
            td_map = {train_model.X: obs, polyak_model.X: obs, A: actions, R: rewards, D: dones, MU: mus, LR: cur_lr}
            print('sess dayo-----: {}'.format(self.sess))
            return names_ops, self.sess.run(run_ops, td_map)[1:] # strip off _train

        def _step(observation, goal, **kwargs):
            return step_model._evaluate([step_model.action, step_model_p], observation, **kwargs)

        def get_actions(o, g, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
            actions, mus = self._step(o, g)
            # states should be dummy states like []
            #TODO Check the dimension of act
            # return act[0], q_val
            return actions, mus


        self.train = train
        # self.save = functools.partial(save_variables, sess=sess, variables=params)
        self.train_model = train_model
        self.step_model = step_model
        self._step = _step
        self.step = self.step_model.step

        def reward_fun(ag_2, g):
            '''
            :param ag_2: shape [nsteps, ag_2_dim]
            :param g: shape [nsteps, g_dim]
            :return:
            '''
            def goal_distance(goal_a, goal_b):
                assert goal_a.shape == goal_b.shape
                return np.linalg.norm(goal_a - goal_b, axis=-1)
            print('ag_2.shape: {}, g.shape: {}'.format(ag_2.shape, g.shape))
            d = goal_distance(ag_2, g)
            print('goal_distance: {}'.format(d))
            distance_threshold = 0.5
            return - (d > distance_threshold).astype(np.float32)

        self.sample_transitions = make_sample_her_transitions(replay_strategy='future', replay_k=4, reward_fun=reward_fun)
        self.buffer = ReplayBuffer(buffer_size, nsteps, self.sample_transitions)

        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)

    def store_episode(self, episode_batch):
        self.buffer.store_episode(episode_batch)

    def sample_batch(self, batch_size):
        '''
        if self.bc_loss:
            transitions = self.buffer.sample(self.batch_size - self.demo_batch_size)
            global demoBuffer
            transitionsDemo = demoBuffer.sample(self.demo_batch_size)
            for k, values in transitionsDemo.items():
                rolloutV = transitions[k].tolist()
                for v in values:
                    rolloutV.append(v.tolist())
                transitions[k] = np.array(rolloutV)
            pass
        else:
        '''
        transitions = self.buffer.sample(batch_size)
        return transitions

class Acer():
    def __init__(self, runner, model, log_interval):
        self.runner = runner
        self.model = model
        self.buffer = model.buffer
        self.log_interval = log_interval
        self.tstart = None
        #self.episode_stats = EpisodeStats(runner.nsteps, runner.nenv)
        self.steps = None

    def call(self, on_policy):
        runner, model, buffer, steps = self.runner, self.model, self.buffer, self.steps
        if on_policy:
            episode = runner.generate_rollouts() #run()
            model.store_episode(episode)
            # obs, actions, rewards, mus, dones, masks = episode
            obs, actions, rewards, mus, dones = episode['o'], episode['u'], episode['r'], episode['mu'], episode['done']
            #if buffer is not None:
            #    buffer.put(obs, actions, rewards, mus, dones, masks)
        else:
            transitions = model.sample_batch(model.nbatch)
            obs, obs_2, rewards, actions, mus, dones = transitions['o'], transitions['o_2'], transitions['r'], transitions['u'], transitions['mu'], transitions['done']
            #obs, actions, rewards, mus, dones, masks = buffer.get()

        # reshape correctly
        print("obs.shape: {}".format(obs.shape))
        print("actions.shape: {}".format(actions.shape))
        print("rewards.shape: {}".format(rewards.shape))
        print("dones.shape: {}".format(dones.shape))
        if on_policy:
            obs = np.squeeze(obs)
            actions = np.squeeze(actions)[:-1]
            rewards = np.squeeze(rewards)[:-1]
            mus = np.squeeze(mus)[:-1]
            dones = np.squeeze(dones)[1:]
        else:
            # Make obs for train from obs and obs_2
            temp_obs = np.expand_dims(np.squeeze(obs_2)[-1], 0)
            print("obs shape: {}, {}, {}".format(np.squeeze(obs).shape, np.squeeze(obs_2)[-1].shape, temp_obs.shape))
            obs = np.concatenate([np.squeeze(obs), temp_obs])
            actions = np.squeeze(actions)
            rewards = np.squeeze(rewards)
            mus = np.squeeze(mus)
            dones = np.squeeze(dones)

        print("obs.shape: {}".format(obs.shape))
        print("actions.shape: {}".format(actions.shape))
        print("rewards.shape: {}".format(rewards.shape))
        print("dones.shape: {}".format(dones.shape))


        '''
        obs = obs.reshape(runner.batch_ob_shape)
        actions = actions.reshape([runner.nbatch])
        rewards = rewards.reshape([runner.nbatch])
        mus = mus.reshape([runner.nbatch, runner.nact])
        dones = dones.reshape([runner.nbatch])
        # masks = masks.reshape([runner.batch_ob_shape[0]])
        '''
        names_ops, values_ops = model.train(obs, actions, rewards, dones, mus, steps)

        if on_policy:
            # records stuff
            pass