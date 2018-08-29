#!/usr/bin/python
import numpy as np
from random import randint
import rospy
from collections import deque
import auto_data_collector.keyboard_encodings as ENC

from rl_msgs.msg import RLStateReward, RLAction
from std_msgs.msg import Int32, UInt32
from common_msgs_gl.msg import PointArray
from utils import convert_episode_to_batch_major

ACT_CMDS = ['up', 'down', 'left', 'right', 'left up', 'left down', 'right up', 'right down', 'stop']

class RolloutWorker:
    def __init__(self, model, dims, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0.0,
                 random_eps=0.0):
        """
        :param policy (class instance): the policy that is used to act
        :param dims (dict of ints): the dimensions for observations (o) and actions (u)
        :param rollout_batch_size (int): the number of parallel rollouts that should be used
        """
        self.model = model; self.dims = dims; self.rollout_batch_size=rollout_batch_size;
        self.exploit = exploit; self.use_target_net = use_target_net; self.compute_Q = compute_Q
        self.noise_eps = noise_eps; self.random_eps = random_eps

        # For initializing hand/object configuration
        self.keyboardDict = ENC.Encodings
        self.selection_choices = self.keyboardDict.keys()  # Note the stop key is index 0
        self.current_selection = 1  # arb assignment
        self.time_taken = 45  # This will give us a fresh start
        self.time_to_live = 45  # Determines how many cycles the movement will occur


        self.reset_all_rollouts()
        rospy.Subscriber('system/state', Int32, self.system_state_callback)
        rospy.Subscriber('rl_env/rl_state_reward', RLStateReward, self.state_reward_callback)
        self.ac_pub = rospy.Publisher('rl_agent/rl_action', RLAction, queue_size=1)

        self.fake_keyboard_pub_ = rospy.Publisher('/keyboard_input', UInt32, queue_size=1)
        # self.initial_config_state_pub = rospy.Publisher('/initial_config_state', PointArray, queue_size=1)
        # self.goal_config_state_pub = rospy.Publisher('/goal_config_state', PointArray, queue_size=1)

    def reset_rollout(self, i):
        """Resets the 'i'-th rollout environment
        """
        self.first = True
        self.current_action = None
        self.current_state = None
        self.achieved_goal = None
        self.current_reward = 0
        self.current_done = 0
        self.current_drop = 1
        self.goal = None
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
        elif len(s) == 21:
            print("contact point correctly incorporated")
            x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, a0, a2, a4, cp_left_x, cp_left_y, cp_right_x, cp_right_y = s
        elif len(s) == 9:
            x0, y0, x2, y2, x4, y4, a0, a2, a4 = s
        if self.dims['o'] == 6:
            s = [x0, y0, x2, y2, x4, y4]
        elif self.dims['o'] == 9:
            # print("a0,a2,a4 in prepare_s: {}, {}, {}".format(a0, a2, a4))
            # x0 /= 100; y0 /= 100; x2 /= 100; y2 /= 100; x4 /= 100; y4 /= 100
            # print("x0: {}".format(x0))
            s = [x0, y0, x2, y2, x4, y4, a0, a2, a4]
        elif self.dims['o'] == 13:
            s = [x0, y0, x2, y2, x4, y4, a0, a2, a4, cp_left_x, cp_left_y, cp_right_x, cp_right_y]
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
            #if self.first:

            self.current_state = self.prepare_s(sr.state)
            self.achieved_goal = self.current_state
            self.current_reward = sr.reward
            print("Current_reward: {}".format(self.current_reward))
            '''
            policy_output = self.model.get_actions(self.current_state, self.goal,
                                                   compute_Q=self.compute_Q,
                                                   noise_eps=self.noise_eps if not self.exploit else 0.,
                                                   random_eps=self.random_eps if not self.exploit else 0.,
                                                   use_target_net=self.use_target_net)
            '''
            actions, mus = self.model._step(self.current_state, self.goal)
            self.current_action = actions[0]
            self.mus = mus
            self.current_done = 0 # dones should come from env
            '''
            if self.compute_Q:
                sampled_action, self.Q = policy_output
                self.current_action = sampled_action # np.argmax(action_vec)
                print("self.current_action: {}".format(self.current_action))
            else:
                #print("policy_output: {}".format(policy_output)) # 2 = left
                sampled_action = policy_output
                self.current_action = sampled_action #np.argmax(action_vec)
            '''
        else:
            return

    def sample_goal(self):
        # check dimensions of goal
        return 0

    def initObjectPosition(self):
        # how to get keyboard stuff
        # self.fake_keyboard_pub_.publish(self.keyboardDict[self.selection_choices[self.current_selection]])
        '''
        print("Keyboard input Down!!!")
        self.fake_keyboard_pub_.publish(self.keyboardDict["KEYX"])
        '''

        print("printing system state: {}".format(self.system_state))
        if self.time_taken >= self.time_to_live:
            self.time_to_live = randint(5, 25)  # between 1.5-2.5 seconds
            self.current_selection = randint(1, len(self.selection_choices) - 1)
            self.time_taken = 0
            new_value = True
 
        self.fake_keyboard_pub_.publish(self.keyboardDict[self.selection_choices[self.current_selection]])
        print("Keyboard input: {}".format(self.selection_choices[self.current_selection]))
        
        self.time_taken = self.time_taken + 1
        #if self.time_taken < 20:
        #    return True
        #else:
        #    return False

    def generate_rollouts(self):
        """Performs 'rollout_batch_size' rollouts in parallel for time horizon 'T' with the current policy
         acting on it accordingly.
         **NOTE** For ModelT42, rollout_batch_size is 1. For simulation, it could be larger than 1.
         """
        self.reset_all_rollouts()
        rate = rospy.Rate(5)  # publishes action every .2 seconds

        # Set /RLAgent/reset true
        rospy.set_param('/RLAgent/reset', 1)
        # TODO: Setting self.initial_o should be handled by the resetter.
        # TODO: Initial_ag should just be initial_o?
        # self.g = self.sample_goal() # I moved this after self.initObjectPosition()
        # TODO: I should replace rospy.set_param with self.sample_goal() very soon.
        # TODO: Check if this works^ <- this is working but I should use ros msg to communicate the reset state in the auto-resetting system.
        #print("RLAgent reset parameter checking: {}".format(rospy.get_param('/RLAgent/reset')))

        # generate episodes
        obs, acts, rewards, values, dones, mus, drops = [], [], [], [], [], [], []
        achieved_goals, successes, goals = [], [], []

        episode = dict(o=None, u=None, r=None, done=None, mu=None, ag=None, drop=None, g=None)
        Qs = []
        init_flag = True
        count_random_action_sent = 0
        while not rospy.is_shutdown():
            # Instead of doing this, I want to get the system_state from ModelT42
            reset_flag = rospy.get_param('/RLAgent/reset')
            #self.initObjectPosition()
            # Let the hand move
            if count_random_action_sent < 10 and self.system_state==2:
                # self.initObjectPosition()
                self.initObjectPosition()
                count_random_action_sent += 1
                if count_random_action_sent == 10:
                    self.fake_keyboard_pub_.publish(self.keyboardDict["KEY_S_"])
                    # initial_config = PointArray()
                    # initial_config.x = []
                    # TODO: How should I generate goals?
                    # TODO: self.goal should be filled here.
                    self.g = self.sample_goal() #TODO: Need to check the dimension of goal
            if reset_flag:
                if self.current_action is not None:  #
                    #print("self.current_state dim : {}, self.current_action dim : {}, self.current_reward.dim : {}".format(
                    #    self.current_state, self.current_action, self.current_reward
                    #))
                    # adding batch dimension for compatibility with OpenAI baselines code
                    obs.append(np.expand_dims(np.array(self.current_state), 0))
                    achieved_goals.append(np.expand_dims(np.array(self.achieved_goal), 0))
                    #TODO: CHange goals to correct ones
                    goals.append(np.expand_dims(np.array(self.achieved_goal), 0))
                    one_hot_action = np.zeros(self.dims['u'])
                    one_hot_action[self.current_action] = 1.0
                    # acts.append(np.expand_dims(one_hot_action, 0))
                    acts.append(np.expand_dims([self.dims['u']], 0))
                    mus.append(self.mus)
                    rewards.append(np.expand_dims(np.array([self.current_reward]), 0))
                    dones.append(np.expand_dims(np.array([self.current_done]), 0))
                    drops.append(np.expand_dims(np.array([self.current_drop]), 0))

                    if self.compute_Q:
                        Qs.append(self.Q)
                    print("current_action!!!!!!!!: {}".format(self.current_action))
                    print("current_action!!!!!!!!: {}".format(one_hot_action))
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
                # episode['o'], episode['u'], episode['r'], episode['v'], episode['d'] = obs, acts, rewards, Qs, dones
                episode['o'], episode['u'], episode['r'], episode['done'] = obs, acts, rewards, dones
                episode['mu'], episode['ag'], episode['drop'], episode['g'] = mus, achieved_goals, drops, goals
                if self.compute_Q:
                    return convert_episode_to_batch_major(episode)  #, np.mean(Qs)
                else:
                    return convert_episode_to_batch_major(episode)


            rate.sleep()
