#!/usr/bin/python
# The code is based off of openai/baselines/her
# https://github.com/openai/baselines/tree/master/baselines/her
from __future__ import print_function

import rospy
import tensorflow as tf
import numpy as np
import os
import math
import datetime

from rl_msgs.msg import RLStateReward, RLAction
from std_msgs.msg import Float32MultiArray, Int32

from util import store_args, convert_episode_to_batch_major
import config

NODE = "RLAgent"
MODE = 'il'  # nfq
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
        # self.num_states = dims['o']

        rospy.Subscriber('system/state', Int32, self.system_state_callback)
        rospy.Subscriber('rl_env/rl_state_reward', RLStateReward, self.state_reward_callback)
        self.ac_pub = rospy.Publisher('rl_agent/rl_action', RLAction, queue_size=1)

    def system_state_callback(self, data):
        self.system_state = data.data

    def state_reward_callback(self, sr):
        if not self.terminated:
            if sr.terminal:
                self.terminated = True
                return
            self.current_state = self.prepare_s(sr.state)
            self.current_reward = sr.reward
            policy_output = self.policy.get_actions(self.current_state,
                                                    compute_Q=self.compute_Q,
                                                    noise_eps=self.noise_eps if not self.exploit else 0.,
                                                    random_eps=self.random_eps if not self.exploit else 0.,
                                                    use_target_net=self.use_target_net)
            if self.compute_Q:
                self.current_action, self.Q = policy_output
            else:
                self.current_action = policy_output
        else:
            return

    def prepare_s(self, s):
        #TODO: refactor this nonsensical function...
        if len(s) == 17:
            x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, a0, a2, a4 = s
        elif len(s) == 9:
            x0, y0, x2, y2, x4, y4, a0, a2, a4 = s
        if self.dims['o'] == 6:
            s = [x0, y0, x2, y2, x4, y4]
        elif self.dims['o'] == 9:
            #print("a0,a2,a4 in prepare_s: {}, {}, {}".format(a0, a2, a4))
            #x0 /= 100; y0 /= 100; x2 /= 100; y2 /= 100; x4 /= 100; y4 /= 100
            #print("x0: {}".format(x0))
            s = [x0, y0, x2, y2, x4, y4, a0, a2, a4]
        else:
            assert False, 'need to set num_states to 6 or 9'
        return s

    def reset_rollout(self, i):
        """Resets the 'i'-th rollout environment
        """
        self.current_action = None
        self.current_state = None
        self.current_reward = 0
        self.Q = None
        self.terminated = False
        self.system_state = None

    def reset_all_rollouts(self):
        """Resets all 'rollout_batch_size' rollout workers
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs 'rollout_batch_size' rollouts in parallel for time horizon 'T' with the current policy
         acting on it accordingly.
         **NOTE** For ModelT42, rollout_batch_size is 1. For simulation, it could be larger than 1.
         """
        self.reset_all_rollouts()
        rate = rospy.Rate(5)  # publishes action every .2 seconds

        # Set /RLAgent/reset true
        rospy.set_param('/RLAgent/reset', "true")
        # TODO: Check if this works^
        print("RLAgent reset parameter checking: {}".format(rospy.get_param('/RLAgent/reset')))

        # generate episodes
        obs, acts, rewards = [], [], []
        episode = dict(o=None, u=None, r=None)
        Qs = []
        while not rospy.is_shutdown():
            # Instead of doing this, I want to get the system_state from ModelT42
            reset_flag = rospy.get_param('/RLAgent/reset')

            if reset_flag:
                if self.current_action is not None:  #
                    print("self.current_state dime : {}, self.current_action dim : {}, self.current_reward.dim : {}".format(
                        self.current_state, self.current_action, self.current_reward
                    ))
                    # adding batch dimension for compatibility with OpenAI baselines code
                    obs.append(np.expand_dims(np.array(self.current_state), 0))
                    # acts.append(np.expand_dims(np.array(self.current_action), 0))
                    one_hot_action = np.zeros(self.dims['u'])
                    one_hot_action[self.current_action] = 1.0
                    acts.append(np.expand_dims(one_hot_action, 0))
                    rewards.append(np.expand_dims(np.array([self.current_reward]), 0))
                    if self.compute_Q:
                        Qs.append(self.Q)

                    rospy.loginfo('RLAgent sampled action: {}'.format(
                            ACT_CMDS[self.current_action]))
                    rla = RLAction(action=self.current_action)
                    print("RLAction msg has been published from ILRL.py!!!")
                    self.ac_pub.publish(rla)
                    # The above line corresponds to Line 284-298 in the original RolloutWorker script from OpenAI
                    # where they do self.env.step(u) kind of thing to get a new state and reward.
                    # In our setting, RLEnv will send back ROS msgs, which will be processed in state_reward_callback.
            #else:
            #    if len(agent.states) > 0:
            #        agent.log_data()
            #        agent.reset()

            # system_state is ready
            if self.system_state == 3 and len(obs) > 0:
                print("System state is ready...return episode, starting a new episode")
                episode['o'], episode['u'], episode['r'] = obs, acts, rewards
                return convert_episode_to_batch_major(episode)
            rate.sleep()


def train(policy, rollout_worker, n_epochs, n_batches):

    for epoch in range(n_epochs):
        episode = rollout_worker.generate_rollouts()
        policy.store_episode(episode)
        for _ in range(n_batches): # update q-values
            policy.train()
        policy.update_target_net() # update the target net less frequently


if __name__ == '__main__':
    rospy.init_node(NODE)
    rospy.loginfo('started RLAgent node')
    policy_type = 'Jun2714152018_eps1_Jun2714312018_eps1_Jul816002018_eps1' #'Jun2714152018_eps1' #'Jul816002018_eps1' #'yutaro' #'
    # NUM_STATES, NUM_ACTIONS, MODEL_NAME = 9, 9, 'il_pol_' + policy_type


    '''
    rl_method = 'il' #'nfq'
    if rl_method == 'il':
        agent = ILRLAgent(NUM_STATES, NUM_ACTIONS, MODEL_NAME)
    elif rl_method == 'nfq':
        NUM_STATES, NUM_ACTIONS = 9, 9
        MODEL_NAME = 'nfq_weights.best.hdf5'
        agent = NFQRLAgent(NUM_STATES, NUM_ACTIONS, MODEL_NAME)
    '''

    #params = config.DEFAULT_PARAMS
    #params = config.prepare_params_mlp(params)
    # configure_ddpg() will return an instance of DDPG class. I should modify the DDPG class
    # so that it follows the ILRLAgent or NFQRLAgent class definition
    # policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return, bc_loss=bc_loss,
    #                                q_filter=q_filter, num_demo=num_demo)


    dims = {'o' : 9, 'u' : 9}
    model_name = 'Jun2714152018_eps1_Jun2714312018_eps1_Jul816002018_eps1'
    n_epochs = 2
    policy = config.configure_mlp(dims=dims, model_name=model_name, model_save_path=MODEL_SAVE_PATH)
    rollout_worker = RolloutWorker(policy, dims)
    train(policy=policy, rollout_worker=rollout_worker, n_epochs=n_epochs)
