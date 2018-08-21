#!/usr/bin/python
import numpy as np
import rospy

from rl_msgs.msg import RLStateReward, RLAction
from std_msgs.msg import Int32
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
            policy_output = self.model.get_actions(self.current_state, self.goal,
                                                   compute_Q=self.compute_Q,
                                                   noise_eps=self.noise_eps if not self.exploit else 0.,
                                                   random_eps=self.random_eps if not self.exploit else 0.,
                                                   use_target_net=self.use_target_net)
            if self.compute_Q:
                sampled_action, self.Q = policy_output
                self.current_action = sampled_action # np.argmax(action_vec)
                print("self.current_action: {}".format(self.current_action))
            else:
                #print("policy_output: {}".format(policy_output)) # 2 = left
                sampled_action = policy_output
                self.current_action = sampled_action #np.argmax(action_vec)
        else:
            return

    def sample_goal(self):
        # check dimensions of goal
        return 0

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
        self.g = self.sample_goal()
        # TODO: I should replace rospy.set_param with self.sample_goal() very soon.
        # TODO: Check if this works^ <- this is working but I should use ros msg to communicate the reset state in the auto-resetting system.
        #print("RLAgent reset parameter checking: {}".format(rospy.get_param('/RLAgent/reset')))

        # generate episodes
        obs, acts, rewards, values = [], [], [], []
        episode = dict(o=None, u=None, r=None, v=None)
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
                episode['o'], episode['u'], episode['r'], episode['v'] = obs, acts, rewards, Qs
                if self.compute_Q:
                    return convert_episode_to_batch_major(episode), np.mean(Qs)
                else:
                    return convert_episode_to_batch_major(episode)
            rate.sleep()