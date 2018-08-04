import numpy as np
import tensorflow as tf
from utils import cat_entropy, mse, find_trainable_variables, Scheduler

from collections import OrderedDict
#from replay_buffer import PrioritizedReplayBuffer
from tensorflow.contrib.staging import StagingArea
from replay_buffer import ReplayBuffer
from util import flatten_grads, import_function

from mpi_adam import MpiAdam
import os

class Model(object):
    def __init__(self, policy, num_states, num_actions, nsteps, nenvs=1,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
                 save_path=None, model_name=None):
        self.save_path = save_path; self.model_name = model_name
        #TODo Figure out what nsteps corresponds to things in mlp.py
        self.sess = tf.get_default_session()
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch]) # action
        ADV = tf.placeholder(tf.float32, [nbatch]) # advantage function
        R = tf.placeholder(tf.float32, [nbatch]) # reward
        LR = tf.placeholder(tf.float32, []) # learning rate

        step_model = policy(self.sess, num_states, num_actions, nenvs, 1, reuse=False)
        train_model = policy(self.sess, num_states, num_actions, nenvs*nsteps, nsteps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train():
            obs, rewards, actions, values = self.episode_batch
            advs = rewards - values
            for step in range(len(obs)):
             cur_lr = lr.value()
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: rewards, LR: cur_lr}
            policy_loss, value_loss, policy_entropy, _ = self.sess.run(
             [pg_loss, vf_loss, entropy, _train],
             td_map
            )
            return policy_loss, value_loss, policy_entropy

        def get_actions(o, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
            act, q_val, _ = step_model.step
            #TODO Check the dimension of act
            return act, q_val

        self.saver = tf.train.Saver()
        self.get_actions = get_actions
        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        tf.global_variables_initializer().run(session=sess)

    def _restore(self, pi_var_list, checkpoint_path):
        saver = tf.train.Saver(var_list=pi_var_list)
        saver.restore(self.sess, checkpoint_path)

    def save_model(self):
        return self.saver.save(self.sess, os.path.join(self.save_path, self.model_name))

    def store_episode(self, episode_batch):
        '''
        :param episode_batch: dict {'o': [1, num_steps, num_state], 'u': [1, num_steps, num_actions],
                                    'v': [1, num_steps, 1], 'r': [1, num_steps, 1]}
        :return:
        '''
        # get obs, rewards, etc to update network
        obs, rewards, actions, values = episode_batch['o'], episode_batch['r'], episode_batch['r'], episode_batch['v']
        self.episode_batch = (obs, rewards, actions, values)


