import tensorflow as tf
import numpy as np
from utils import cat_entropy, mse, find_trainable_variables, Scheduler
from utils import get_by_index, q_explained_variance
from utils import cat_entropy_softmax, check_shape, batch_to_seq, seq_to_batch

from collections import deque

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
    def __init__(self, policy, num_states, num_actions, nenvs, nsteps, num_rollouts,
                 ent_coef, q_coef, gamma, max_grad_norm, lr,
                 rprop_alpha, rprop_epsilon, total_timesteps, lrschedule,
                 c, trust_region, alpha, delta,
                 buffer_size, bc_loss):
        self.bc_loss = bc_loss
        self.sess = sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession() # what is this for?
        self.nenvs = nenvs; self.nsteps = nsteps; self.num_rollouts = num_rollouts
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
        step_ob_placeholder = tf.placeholder(dtype=tf.int32, shape=(1, 13 + 13)) # change this to shape=(nenvs,13 + 13)
        train_ob_placeholder = tf.placeholder(dtype=tf.int32, shape=(nenvs*(nsteps+1), 13 + 13))  # change this to shape=(nenvs*(nsteps+1), 13 + 13)
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
        v = tf.reduce_sum(train_model_p * train_model.q, axis=-1) # shape is [nenvs * (nsteps + 1)]

        print("train_model_p shape: {}".format(train_model_p.get_shape().as_list()))
        print("v shape: {}".format(v.get_shape().as_list()))
        print("polyak_model_p shape: {}".format(polyak_model_p.get_shape().as_list()))
        print("train_model.q shape: {}".format(train_model.q.get_shape().as_list()))

        # strip off last step # I'm assuming that the reason you need nsteps+1 for train_model is to get obs_{t+1} info
        # for Experience Replay (each tuple in the buffer should be (o_t, o_{t+1}, r_t, a_t)
        # The above interpretation (each tuple..) is false for our setting since it's trajecotry based; should be updated when I have time.
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
            #TODO: Implement trust_region method if I want to use it
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
            # TODO: Implement trust_region method if I want to use it
            run_ops = run_ops + []
            names_ops = names_ops + []

        def train(obs, goals, actions, rewards, dones, mus, steps):
            cur_lr = lr.value_steps(steps)
            obs_goals_concat = np.concatenate((obs, goals), axis=1)
            print("Right before feeding: obs_goals_concat.shape: {}".format(obs_goals_concat.shape))
            td_map = {train_model.X: obs_goals_concat, polyak_model.X: obs_goals_concat, A: actions, R: rewards, D: dones, MU: mus, LR: cur_lr}
            #print('sess dayo-----: {}'.format(self.sess))
            return names_ops, self.sess.run(run_ops, td_map)[1:]  # strip off _train

        def _step(obs, goal, **kwargs):
            obs = np.array(obs)
            goal = np.array(goal)
            print("In rollout.py _step function: obs.shape: {}, goal.shape: {}".format(obs.shape, goal.shape))
            observation = np.concatenate((obs, goal))
            print("In rollout.py _step function: observation.shape: {}".format(observation.shape))
            return step_model._evaluate([step_model.action, step_model_p], observation, **kwargs)

        def get_actions(o, g, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
            # TODO: I don't think this function is used anymore?
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

        def reward_fun(ag, g):
            '''
            :param ag_2: shape [nsteps, ag_2_dim]
            :param g: shape [nsteps, g_dim]
            :return:
            '''
            def goal_distance(goal_a, goal_b):
                assert goal_a.shape == goal_b.shape
                return np.linalg.norm(goal_a - goal_b, axis=-1)
            print('ag_2.shape: {}, g.shape: {}'.format(ag.shape, g.shape))
            d = goal_distance(ag, g)
            print('goal_distance: {}'.format(d))
            distance_threshold = 0.5
            return - (d > distance_threshold).astype(np.float32)

        self.sample_transitions = make_sample_her_transitions(replay_strategy='future', replay_k=4, reward_fun=reward_fun)
        self.buffer = ReplayBuffer(buffer_size, num_rollouts, self.sample_transitions)

        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)

    def store_episode(self, episode_batch):
        self.buffer.store_episode(episode_batch)

    def sample_batch(self, batch_size, nsteps):
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
        transitions = self.buffer.sample(batch_size, nsteps)
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
        self.success_history = deque()
        self.reward_history = deque()

    def sample_trajectory_on_policy(self, episode_batch, nsteps):
        nsteps += 1
        # Shape: i.e. want to convert (50,13) into shape (41,13)
        # TODO: Need to think about this carefuly. It's related to HER sampling too.
        # where is the drop time step?
        if np.all(np.squeeze(episode_batch['drop'])==0) and np.all(np.squeeze(episode_batch['stuck'])==0): # checking if every element is zero
            print("Neither drop nor stuck didn't happen")
            drop_time_id = self.model.num_rollouts # getting the index for HER drop_time_steps
        else:
            if not np.all(np.squeeze(episode_batch['drop'])==0):
                print("drop happened")
                drop_time_id = np.where(np.squeeze(episode_batch['drop'])==1)[0][0]
            if not np.all(np.squeeze(episode_batch['stuck'])==0):
                print("stuck happened")
                drop_time_id = np.where(np.squeeze(episode_batch['stuck'])==1)[0][0]
        print("On-Policy drop_time_id: {}".format(drop_time_id))
        print("nsteps: {}".format(nsteps))
        if drop_time_id <= nsteps:
            t_sample = nsteps
        else:
            t_sample = np.random.randint(low=nsteps, high=drop_time_id)
        print("t_samples: {}".format(t_sample))
        print("episode_batch['o'].shape: {}".format(episode_batch['o'].shape))
        temp_o = episode_batch['o'][:,(t_sample-nsteps):t_sample,:]
        temp_g = episode_batch['g'][:, (t_sample - nsteps):t_sample, :]
        temp_u = episode_batch['u'][:,(t_sample-nsteps):t_sample,:]
        temp_r = episode_batch['r'][:,(t_sample-nsteps):t_sample,:]
        temp_mu = episode_batch['mu'][:,(t_sample-nsteps):t_sample,:]
        temp_done = episode_batch['done'][:,(t_sample-nsteps):t_sample,:]

        print("temp_o.shape: {}".format(temp_o.shape))
        return temp_o, temp_g, temp_u, temp_r, temp_mu, temp_done


    def call(self, on_policy):
        runner, model, buffer, steps = self.runner, self.model, self.buffer, self.steps
        n_state_space = 13
        if on_policy:
            i = 0
            obs = np.empty((model.nenvs, model.nsteps+1, n_state_space)) # Need to reshape right before model.train
            goals = np.empty((model.nenvs, model.nsteps+1, n_state_space))
            actions = np.empty((model.nenvs, model.nsteps+1, 1))
            rewards = np.empty((model.nenvs, model.nsteps + 1, 1))
            mus = np.empty((model.nenvs, model.nsteps + 1, 9))
            dones = np.empty((model.nenvs, model.nsteps + 1, 1))
            while i < model.nenvs:
                print("ON_POLICY {}/{} th ROLLOUT STARTING..........".format(i + 1, model.nenvs))
                episode = runner.generate_rollouts() #run()
                print("printing episode['drop'][0][0]: {}".format(episode['drop'][0][0]))
                print("printing episode['stuck'][0][0]: {}".format(episode['stuck'][0][0]))
                if episode['drop'][0][0][0] == 1 or episode['stuck'][0][0][0] == 1:
                    print("Drop happened at time step 1. Ignoring this episode...")
                    return
                model.store_episode(episode) #TODO: Fix the length of batch shape in replay_buffer_her.py
                # obs, actions, rewards, mus, dones, masks = episode
                # I have to make sure that I "sample" the trajectory of length nsteps from the episode, whose length is num_rollouts.
                # TODO : The following line will throw an error for inconsistent shape
                obs[i], goals[i], actions[i], rewards[i], mus[i], dones[i] = self.sample_trajectory_on_policy(episode, model.nsteps)
                # obs[i], actions[i], rewards[i], mus[i], dones[i] = episode['o'], episode['u'], episode['r'], episode['mu'], episode['done']
                i += 1
            # Record stats: this is assuming that I fill up dones and rewards till the end even if rollout terminates earlier.
            self.success_history.append(np.mean(dones[:, -1]))
            self.reward_history.append(np.mean(rewards[:, -1]))
            # TODO: Decide what threshold I should use to determine "success" for each episode. See self.dones in state_reward_callback in rollout.py
            # ToDO: Specifically, finish self.is_success() function.
        else:
            transitions = model.sample_batch(model.nenvs, model.nsteps)
            # I shouldn't have any o_2 thing here. It's some old stuff from HER+DDPG
            # obs, obs_2, rewards, actions, mus, dones = transitions['o'], transitions['o_2'], transitions['r'], transitions['u'], transitions['mu'], transitions['done']
            obs, goals, rewards, actions, mus, dones = transitions['o'], transitions['g'], transitions['r'], transitions['u'], transitions['mu'], transitions['done']
            #obs, actions, rewards, mus, dones, masks = buffer.get()

        # reshape correctly
        print("obs.shape: {}".format(obs.shape))
        print("goals.shape: {}".format(goals.shape))
        print("actions.shape: {}".format(actions.shape))
        print("rewards.shape: {}".format(rewards.shape))
        print("dones.shape: {}".format(dones.shape))
        if on_policy:
            obs = obs.reshape(model.nenvs*(model.nsteps+1), n_state_space)
            goals = goals.reshape(model.nenvs*(model.nsteps+1), n_state_space)
            actions = np.squeeze(actions[:, :-1, :].reshape(model.nenvs*model.nsteps, -1))
            rewards = np.squeeze(rewards[:, 1:, :].reshape(model.nenvs*model.nsteps, -1))
            mus = mus[:, 1:, :].reshape(model.nenvs*model.nsteps, -1)
            dones = np.squeeze(dones[:, 1:, :].reshape(model.nenvs*model.nsteps, -1))
        else:
            '''
            # Make obs for train from obs and obs_2
            temp_obs = np.expand_dims(np.squeeze(obs_2)[-1], 0)
            print("obs shape: {}, {}, {}".format(np.squeeze(obs).shape, np.squeeze(obs_2)[-1].shape, temp_obs.shape))
            obs = np.concatenate([np.squeeze(obs), temp_obs])
            actions = np.squeeze(actions)
            rewards = np.squeeze(rewards)
            mus = np.squeeze(mus)
            dones = np.squeeze(dones)
            '''
            obs = obs.reshape(model.nenvs*(model.nsteps+1), n_state_space)
            goals = goals.reshape(model.nenvs*(model.nsteps+1), n_state_space)
            actions = np.squeeze(actions[:, :-1, :].reshape(model.nenvs*model.nsteps, -1))
            rewards = np.squeeze(rewards[:, 1:, :].reshape(model.nenvs*model.nsteps, -1))
            mus = mus[:, 1:, :].reshape(model.nenvs*model.nsteps, -1)
            dones = np.squeeze(dones[:, 1:, :].reshape(model.nenvs*model.nsteps, -1))

        print("obs.shape: {}".format(obs.shape))
        print("goals.shape: {}".format(goals.shape))
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
        names_ops, values_ops = model.train(obs, goals, actions, rewards, dones, mus, steps)

        if on_policy:
            # records stuff
            pass