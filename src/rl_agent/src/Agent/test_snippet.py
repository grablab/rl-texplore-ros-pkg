#!/usr/bin/python
# The code is based off of openai/baselines/her
# https://github.com/openai/baselines/tree/master/baselines/her
from __future__ import print_function

import rospy
import tensorflow as tf
import numpy as np
from collections import deque
import os
import math
import datetime

from rl_msgs.msg import RLStateReward, RLAction
from std_msgs.msg import Float32MultiArray, Int32

from util import store_args, convert_episode_to_batch_major
import config

NODE = "RLAgent"
MODE = 'actor_critic_mlp' # 'il'  # nfq
MODEL_SAVE_PATH = os.path.join(os.environ['HOME'], 'grablab-ros/src/projects/sliding_policies/models/' + MODE + '/')
ACT_CMDS = ['up', 'down', 'left', 'right', 'left up', 'left down', 'right up', 'right down', 'stop']


class RolloutWorker:
    def __init__(self, policy, dims, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0):
        """
        :param policy (class instance): the policy that is used to act
        :param dims (dict of ints): the dimensions for observations (o) and actions (u)
        :param rollout_batch_size (int): the number of parallel rollouts that should be used
        """
        self.policy = policy; self.dims = dims; self.rollout_batch_size=rollout_batch_size;
        self.exploit = exploit; self.use_target_net = use_target_net; self.compute_Q = compute_Q
        self.noise_eps = noise_eps; self.random_eps = random_eps

        self.reset_all_rollouts()
        rospy.Subscriber('system/state', Int32, self.system_state_callback)
        rospy.Subscriber('rl_env/rl_state_reward', RLStateReward, self.state_reward_callback)
        self.ac_pub = rospy.Publisher('rl_agent/rl_action', RLAction, queue_size=1)

    def reset_rollout(self, i):
        """Resets the 'i'-th rollout environment
        """
        self.current_action = None
        self.current_state = None
        self.current_reward = 0
        self.Q = None
        self.terminated = False
        self.system_state = None
        self.Q_history = deque()

    def reset_all_rollouts(self):
        """Resets all 'rollout_batch_size' rollout workers
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def prepare_s(self, s):
        # TODO: refactor this nonsensical function...
        if len(s) == 17:
            x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, a0, a2, a4 = s
        elif len(s) == 9:
            x0, y0, x2, y2, x4, y4, a0, a2, a4 = s
        if self.dims['o'] == 6:
            s = [x0, y0, x2, y2, x4, y4]
        elif self.dims['o'] == 9:
            # print("a0,a2,a4 in prepare_s: {}, {}, {}".format(a0, a2, a4))
            # x0 /= 100; y0 /= 100; x2 /= 100; y2 /= 100; x4 /= 100; y4 /= 100
            # print("x0: {}".format(x0))
            s = [x0, y0, x2, y2, x4, y4, a0, a2, a4]
        else:
            assert False, 'need to set num_states to 6 or 9'
        return s

    def system_state_callback(self, data):
        self.system_state = data.data

    def state_reward_callback(self, sr):
        if not self.terminated:
            if sr.terminal:
                self.terminated = True
                return
            self.current_state = self.prepare_s(sr.state)
            self.current_reward = sr.reward
            print("Current_reward: {}".format(self.current_reward))
            policy_output = self.policy.get_actions(self.current_state,
                                                    compute_Q=self.compute_Q,
                                                    noise_eps=self.noise_eps if not self.exploit else 0.,
                                                    random_eps=self.random_eps if not self.exploit else 0.,
                                                    use_target_net=self.use_target_net)
            if self.compute_Q:
                sampled_action, self.Q = policy_output
                self.current_action = sampled_action # np.argmax(action_vec)
            else:
                #print("policy_output: {}".format(policy_output)) # 2 = left
                sampled_action = policy_output
                self.current_action = sampled_action #np.argmax(action_vec)
        else:
            return

    def generate_rollouts(self):
        """Performs 'rollout_batch_size' rollouts in parallel for time horizon 'T' with the current policy
         acting on it accordingly.
         **NOTE** For ModelT42, rollout_batch_size is 1. For simulation, it could be larger than 1.
         """
        self.reset_all_rollouts()
        rate = rospy.Rate(5)  # publishes action every .2 seconds

        # Set /RLAgent/reset true
        rospy.set_param('/RLAgent/reset', 1)
        # TODO: Check if this works^ <- this is working but I should use ros msg to communicate the reset state in the auto-resetting system.
        #print("RLAgent reset parameter checking: {}".format(rospy.get_param('/RLAgent/reset')))

        # generate episodes
        obs, acts, rewards = [], [], []
        episode = dict(o=None, u=None, r=None)
        Qs = []
        while not rospy.is_shutdown():
            # Instead of doing this, I want to get the system_state from ModelT42
            reset_flag = rospy.get_param('/RLAgent/reset')

            if reset_flag:
                if self.current_action is not None:  #
                    #print("self.current_state dim : {}, self.current_action dim : {}, self.current_reward.dim : {}".format(
                    #    self.current_state, self.current_action, self.current_reward
                    #))
                    # adding batch dimension for compatibility with OpenAI baselines code
                    obs.append(np.expand_dims(np.array(self.current_state), 0))
                    one_hot_action = np.zeros(self.dims['u'])
                    one_hot_action[self.current_action] = 1.0
                    acts.append(np.expand_dims(one_hot_action, 0))
                    rewards.append(np.expand_dims(np.array([self.current_reward]), 0))

                    if self.compute_Q:
                        Qs.append(self.Q)

                    rospy.loginfo('RLAgent sampled action: {}'.format(
                            ACT_CMDS[self.current_action]))
                    rla = RLAction(action=self.current_action)
                    #print("RLAction msg has been published from ILRL.py!!!")
                    self.ac_pub.publish(rla)
                    # The above line corresponds to Line 284-298 in the original RolloutWorker script from OpenAI
                    # where they do self.env.step(u) kind of thing to get a new state and reward.
                    # In our setting, RLEnv will send back ROS msgs, which will be processed in state_reward_callback.
            #else:
            #    if len(agent.states) > 0:
            #        agent.log_data()
            #        agent.reset()

            # system_state is ready for the next episode
            if self.system_state == 3 and len(obs) > 0:
                print("System state is ready...return episode, starting a new episode")
                episode['o'], episode['u'], episode['r'] = obs, acts, rewards
                if self.compute_Q:
                    return convert_episode_to_batch_major(episode), np.mean(Qs)
                else:
                    return convert_episode_to_batch_major(episode)
            rate.sleep()


def train(policy, rollout_worker, n_epochs, n_batches):
    Q_history = deque()
    q_hist, critic_loss_hist, actor_loss_hist = [], [], []
    for epoch in range(n_epochs):
        #print('ok')
        if rollout_worker.compute_Q:
            episode, mean_Q = rollout_worker.generate_rollouts()
        else:
            episode = rollout_worker.generate_rollouts()
        # TODO Check how store_episode will go
        policy.store_episode(episode)
        critic_loss_que, actor_loss_que = [], []
        for i in range(n_batches): # update q-values
            critic_loss, actor_loss = policy.train()
            critic_loss_que.append(critic_loss); actor_loss_que.append(actor_loss)
            # print("n_batch: {}, critic_loss: {}, actor_loss: {}".format(i, critic_loss, actor_loss))
        print("Mean Q-value: {}".format(mean_Q))
        mean_critic_loss = np.mean(critic_loss_que)
        mean_actor_loss = np.mean(actor_loss_que)
        print("Mean critic loss: {}".format(mean_critic_loss))
        print("Mean actor loss: {}".format(mean_actor_loss))
        q_hist.append(mean_Q)
        critic_loss_hist.append(mean_critic_loss)
        actor_loss_hist.append(mean_actor_loss)
        policy.update_target_net() # update the target net less frequently
        np.save('/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/results/q_val.npy', np.array(q_hist))
        np.save('/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/results/cri_loss.npy', np.array(critic_loss_hist))
        np.save('/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/results/actor_loss.npy', np.array(actor_loss_hist))
        save_loc = policy.save_model()
        print('saved model at : {} after {} epochs'.format(save_loc, epoch+1))

if __name__ == '__main__':
    rospy.init_node(NODE)
    rospy.loginfo('started RLAgent node')
    dims = {'o': 9, 'u': 9}
    model_name = 'Jun2714152018_eps1_Jun2714312018_eps1_Jul816002018_eps1'
    n_epochs = 100000
    random_eps = 0.1
    policy = config.configure_mlp(dims=dims, model_name=model_name, model_save_path=MODEL_SAVE_PATH)
    print(policy)
    rollout_worker = RolloutWorker(policy, dims, use_target_net=True, compute_Q=True, random_eps=random_eps)
    train(policy=policy, rollout_worker=rollout_worker, n_epochs=n_epochs, n_batches=100)
