#!/usr/bin/python
import numpy as np
from random import randint
import rospy
from collections import deque
import auto_data_collector.keyboard_encodings as ENC

from sklearn.neighbors import NearestNeighbors

import time
import glob
from pandas import DataFrame

from std_srvs.srv import SetBool
from rl_msgs.msg import RLStateReward, RLAction
from common_msgs_gl.srv import SendInt
from std_msgs.msg import Int32, UInt32, Int32MultiArray
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray
from marker_tracker.msg import ImageSpacePoseMsg
from utils import convert_episode_to_batch_major

ACT_CMDS = ['up', 'down', 'left', 'right', 'left up', 'left down', 'right up', 'right down', 'stop']
ACT_CMDS_DIC = {'up':0, 'down':1, 'left':2, 'right':3, 'left up':4, 'left down':5, 'right up':6, 'right down':7, 'stop':8}

########## config for object resetter ############

initialize_hand_srv_ = rospy.ServiceProxy('/manipulation_manager/set_mode', SendInt)
reset_obj_srv_ = rospy.ServiceProxy('/system_reset/reset_object', SendInt)

fake_keyboard_pub_ = rospy.Publisher('/keyboard_input', UInt32, queue_size=1)

RAISE_CRANE_RESET =0
LOWER_CRANE_OBJECT_READY =1

GET_GRIPPER_READY = 0
ALLOW_GRIPPER_TO_MOVE =1
OPEN_GRIPPER =2

seconds_until_next_reset = 8
####################################################


class RolloutWorker:
    def __init__(self, model, dims, num_rollouts, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0.0,
                 random_eps=0.0, marker_space_markers=[0,1,2,3,4,5,6,7], record_demo_data_w_keyboard=False):
        """
        :param policy (class instance): the policy that is used to act
        :param dims (dict of ints): the dimensions for observations (o) and actions (u)
        :param rollout_batch_size (int): the number of parallel rollouts that should be used
        """
        self.first = True
        self.record_demo_data_w_keyboard = record_demo_data_w_keyboard
        if self.record_demo_data_w_keyboard:
            self.keyboard_input = 'stop'
            rospy.Subscriber('/keyboard_input', UInt32, self.keyboard_callback)
            self.contact_point_object_left = None
            self.contact_point_object_right = None
            rospy.Subscriber('contact_point_detector/contact_point_object_left', Int32MultiArray, self.contact_point_object_left_callback)
            rospy.Subscriber('contact_point_detector/contact_point_object_right', Int32MultiArray, self.contact_point_object_right_callback)

        self.num_rollouts = num_rollouts
        self.model = model; self.dims = dims; self.rollout_batch_size=rollout_batch_size;
        self.exploit = exploit; self.use_target_net = use_target_net; self.compute_Q = compute_Q
        self.noise_eps = noise_eps; self.random_eps = random_eps
        self.marker_space_markers = marker_space_markers
        self.current_config = None

        # Goal sampling
        self.prepare_for_goal_sampling()

        # Stats
        self.success_history = deque()

        # For initializing hand/object configuration
        self.keyboardDict = ENC.Encodings
        self.selection_choices = self.keyboardDict.keys()  # Note the stop key is index 0
        self.current_selection = 1  # arb assignment
        self.time_taken = 45  # This will give us a fresh start
        self.time_to_live = 45  # Determines how many cycles the movement will occur

        # These will be enables later
        self.enable = False
        self.enable_fake_keyboard_ = False
        self.master_tic = time.time()
        self.initialized = False

        # Determine if any modes are occuring
        self.is_dropped_ = False
        self.is_stuck_ = False

        # Count how many of each occured
        self.num_normal_ = 0
        self.num_dropped_ = 0
        self.num_stuck_ = 0
        self.num_sliding = 0

        # This is used to reset the motors if something bad occurs
        self.load_history_ = []
        self.load_history_length_ = 5
        self.reset_to_save_motors = False
        self.single_motor_save_ = []

        self.recent_object_move_dist_ = 50  # Arbritrary value to start


        self.reset_all_rollouts()
        rospy.Subscriber('system/state', Int32, self.system_state_callback)
        rospy.Subscriber('rl_env/rl_state_reward', RLStateReward, self.state_reward_callback)
        rospy.Subscriber("/marker_tracker/image_space_pose_msg", ImageSpacePoseMsg, self.marker_tracker_callback)

        rospy.Subscriber("/stuck_drop_detector/object_dropped_msg", Bool, self.itemDroppedCallback)
        rospy.Subscriber("/stuck_drop_detector/object_stuck_msg", Bool, self.itemStuckCallback)
        rospy.Subscriber("/stuck_drop_detector/object_move_dist", Int32, self.objectMoveDist)
        rospy.Subscriber("/gripper/load", Float32MultiArray, self.gripperLoadCallback)

        self.init_config_pub_ = rospy.Publisher("rl_agent/initial_config_state", ImageSpacePoseMsg, queue_size=1)
        self.goal_config_pub_ = rospy.Publisher("rl_agent/goal_config_state", ImageSpacePoseMsg, queue_size=1)
        self.ac_pub = rospy.Publisher('rl_agent/rl_action', RLAction, queue_size=1)
        self.fake_keyboard_pub_ = rospy.Publisher('/keyboard_input', UInt32, queue_size=1)


        ### The hand should grasp the object firmly when enable_data_save() is called ###
        time.sleep(5)
        print("Opening the gripper")
        initialize_hand_srv_(OPEN_GRIPPER)  # let the system stop
        time.sleep(8)
        print("RAISE_CRANE_RESET")
        reset_obj_srv_(RAISE_CRANE_RESET)
        time.sleep(5)
        print("Grasping the object...")
        initialize_hand_srv_(GET_GRIPPER_READY)
        time.sleep(8)
        print("lowering crane object ready")
        reset_obj_srv_(LOWER_CRANE_OBJECT_READY)
        time.sleep(2)

        ### Send True from terminal to start RL training.
        enable_srv_ = rospy.Service("/auto_data_collector/enable_collection", SetBool, self.enable_data_save)

        print('self.model.nbatch: {}'.format(self.model.nbatch))

    def enable_data_save(self, req):
        self.enable = req.data
        if self.enable == True:
            self.initialized = True

        return [self.enable, "Successfully changed enable bool"]

    def marker_tracker_callback(self, data):
        ids = sorted(self.marker_space_markers + [8, 9]) # [8,9] are for contact points
        x = [0 for _ in range(len(ids))]
        y = [0 for _ in range(len(ids))]
        angles = [0. for _ in range(len(ids))]
        for id_, pos_x, pos_y, angle in zip(data.ids, data.posx, data.posy, data.angles):
            if id_ not in self.marker_space_markers:
                continue
            x[id_] = pos_x
            y[id_] = pos_y
            angles[id_] = angle
        x[8] = self.contact_point_object_left[0]; x[9] = self.contact_point_object_right[0]
        y[8] = self.contact_point_object_left[1]; y[9] = self.contact_point_object_right[1]
        angles[8] = 0; angles[9] = 0
        ispm = ImageSpacePoseMsg()
        ispm.ids, ispm.posx, ispm.posy, ispm.angles = ids, x, y, angles
        self.current_config = ispm

    def reset_rollout(self, i):
        """Resets the 'i'-th rollout environment
        """
        self.first = True
        self.current_action = None
        self.current_state = None
        self.achieved_goal = None
        self.current_reward = 0
        self.current_done = 0
        self.current_drop = 0
        self.current_stuck = 0
        self.goal = None
        self.Q = None
        self.terminated = False
        # self.system_state =
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
        print("system_state callback: {}".format(data.data))
        self.system_state = data.data

    def is_success(self, obs, goal):
        # go through each value of obs and goal to determine success
        obs = np.array(obs)
        goal = np.array(goal)
        print("In is_success, obs.shape: {}, goal.shape: {}".format(obs.shape, goal.shape))
        pos_diff = np.linalg.norm(obs[:6]-goal[:6])
        angle_diff = np.linalg.norm(obs[6:9]-goal[6:9])
        contact_point_diff = np.linalg.norm(obs[9:13]-goal[9:13])
        # TODO : I think I should have a different threshold for each pos/angle/contact_point_diff to determine success
        # TODo: Make sure that I fill up the succcess array (or done array) with 1s if it returns 1 earlier than the end of the episode.
        return 0

    def state_reward_callback(self, sr):
        print("Is this called at all?????????????????????????????")
        if not self.terminated:
            if sr.terminal:
                self.terminated = True
                return

            self.current_state = self.prepare_s(sr.state)
            print("self.current_state: {}".format(self.current_state))
            self.achieved_goal = self.current_state
            self.current_reward = sr.reward
            print("Current_reward: {}".format(self.current_reward))
            if self.first:
                self.goal = self.current_state
                self.first = False
            '''
            policy_output = self.model.get_actions(self.current_state, self.goal,
                                                   compute_Q=self.compute_Q,
                                                   noise_eps=self.noise_eps if not self.exploit else 0.,
                                                   random_eps=self.random_eps if not self.exploit else 0.,
                                                   use_target_net=self.use_target_net)
            '''
            actions, mus = self.model._step(self.current_state, self.goal)
            print("in state_reward_callback, actions[0]: {}".format(actions[0]))
            self.current_action = actions[0]
            self.mus = mus
            self.current_done = self.is_success(self.current_state, self.goal) # 0 # dones should come from env
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

    def prepare_for_goal_sampling(self):
        def get_init_state(episode):
            mean = episode.iloc[:, :-1].mean()
            first_5 = episode.iloc[:5, :-1]
            min_id = abs(first_5 - mean).sum(axis=1).idxmin()
            #print(min_id)
            init_config = episode.iloc[min_id, :-1]
            # print(init_config)
            return init_config

        def get_goal_state(episode):
            mean = episode.iloc[:, :-1].mean()
            last_5 = episode.iloc[-5:, :-1]
            min_id = abs(last_5 - mean).sum(axis=1).idxmin()
            #print(min_id)
            #print(len(episode))
            goal_config = episode.iloc[min_id, :-1]
            # print(goal_config)
            return goal_config
        n_files = len(glob.glob("/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/data-init-goal/keyboard_demo_20180907/*"))
        self.init_config_mat = np.empty([n_files, 13])
        self.init_goal_pair_list = []
        # for i in sorted(os.listdir(data_dir)):
        for i, f in enumerate(
                sorted(glob.glob("/home/grablab/grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/data-init-goal/keyboard_demo_20180907/*"), key=lambda s: int(s.split('_', 4)[-1].split('.')[0]))):
            # print(i)
            # f = os.path.join(data_dir, i)
            a = DataFrame(np.loadtxt(f, delimiter=','))
            # deleting rows with all zeros
            df = a[(a.T != 0).any()]
            df = df.drop(df.index[(
                        df[2].eq(0) | df[3].eq(0) | df[0].eq(0) | df[1].eq(0) | df[4].eq(0) | df[5].eq(0) | df[9].eq(
                    0) | df[10].eq(0) | df[11].eq(0) | df[12].eq(0))])
            df.index = range(len(df))

            # just add init and goal setter here
            init_state = get_init_state(df)
            goal_state = get_goal_state(df)
            self.init_goal_pair_list.append((init_state, goal_state))
            self.init_config_mat[i, :] = init_state

        #self.init_config_mat =  # get this from a file
        #self.init_goal_pair_list = # get also this from a file
        print("init_config_mat is prepared. shape: {}, last row: {}".format(self.init_config_mat.shape, self.init_config_mat[n_files-1,:]))
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.init_config_mat)

    def prepare_init_goal_state(self, config_msg):
        # config_msg : ImageSpacePoseMsg
        init_config = [config_msg.posx[0], config_msg.posy[0], config_msg.posx[2], config_msg.posy[2], config_msg.posx[4], config_msg.posy[4],
        config_msg.angles[0], config_msg.angles[2], config_msg.angles[4], config_msg.posx[8], config_msg.posy[8], config_msg.posx[9], config_msg.posy[9]]
        return init_config

    def list2imgmsg(self, list_data):
        ispm = ImageSpacePoseMsg()
        ids = sorted(self.marker_space_markers + [8, 9]) # [8,9] are for contact points
        x = [0 for _ in range(len(ids))]
        y = [0 for _ in range(len(ids))]
        angles = [0. for _ in range(len(ids))]
        x[0] = list_data[0]; x[2] = list_data[2]; x[4] = list_data[4]
        y[0] = list_data[1]; y[2] = list_data[3]; y[4] = list_data[5]
        angles[0] = list_data[6]; angles[2] = list_data[7]; angles[4] = list_data[8]
        x[8] = list_data[9]; y[8] = list_data[10]
        x[9] = list_data[11]; y[9] = list_data[12]
        ispm.ids, ispm.posx, ispm.posy, ispm.angles = ids, x, y, angles
        return ispm

    def sample_goal(self, init_config):
        # init_config: ImageSpacePoseMsg
        # check dimensions of goal
        # goal_shape = init_config.shape # Can't do this, init_config
        print("What's inside init_config_msg?: {}".format(init_config))
        # The kneighbors expects init_config as numpy 2D array
        #init_config = self.prepare_config_state(init_config_msg)

        distances, indices = self.nbrs.kneighbors(np.array(init_config).reshape(1,-1))
        idx = int(np.squeeze(indices))
        goal_config = self.init_goal_pair_list[idx][1] # it's a pair (init, goal)
        #TODO: Change init_config to goal_config once I finish prepare_config_state
        return goal_config

    def init_object_position(self):
        # how to get keyboard stuff
        # self.fake_keyboard_pub_.publish(self.keyboardDict[self.selection_choices[self.current_selection]])
        '''
        print("Keyboard input Down!!!")
        self.fake_keyboard_pub_.publish(self.keyboardDict["KEYX"])
        '''
        print("Initializing the hand/object configuration...")
        print("printing system state: {}".format(self.system_state))
        if self.time_taken >= self.time_to_live:
            self.time_to_live = randint(5, 25)  # between 1.5-2.5 seconds
            while True:
                self.current_selection = randint(1, len(self.selection_choices) - 1)
                if self.selection_choices[self.current_selection] not in {'KEYX', 'KEYXX', 'KEYXXX', 'KEY_S_', 'KEYW', 'KEYE', 'KEYQ'}:
                    break
            self.time_taken = 0
            new_value = True

        if self.selection_choices[self.current_selection] in {'KEYX', 'KEYXX', 'KEYXXX', 'KEY_S_', 'KEYW', 'KEYE', 'KEYQ'}:
            pass
        else:
            self.fake_keyboard_pub_.publish(self.keyboardDict[self.selection_choices[self.current_selection]])
            print("Keyboard input: {}".format(self.selection_choices[self.current_selection]))
            self.count_random_action_sent += 1

        if self.is_dropped_ or self.is_stuck_ or self.reset_to_save_motors == True:
            self.resetObject()

        self.time_taken = self.time_taken + 1
        # rospy.spin_once()  # allow current_config to be updated by callback

        #if self.time_taken < 20:
        #    return True
        #else:
        #    return False

    def contact_point_object_left_callback(self, data):
        self.contact_point_object_left = data.data

    def contact_point_object_right_callback(self, data):
        self.contact_point_object_right = data.data

    def generate_rollouts_w_keyboard(self):
        # generate episodes
        obs, acts, rewards, values, dones, mus, drops = [], [], [], [], [], [], []
        achieved_goals, goals, stucks = [], [], []

        rate = rospy.Rate(5)

        episode = dict(o=None, u=None, drop=None, stuck=None)
        self.count_random_action_sent = 0
        while not rospy.is_shutdown():
            # Let the hand move
            #print("system_state: {}, self.enable: {}, self.initialized: {}, self.count_random_action_sent: {}".format(self.system_state, self.enable, self.initialized, self.count_random_action_sent))
            if self.count_random_action_sent < 5 and self.system_state==2:
                if self.count_random_action_sent == 1:
                    self.g = self.current_config
                self.init_object_position()
                if self.count_random_action_sent == 5:
                    print("Initialization Done")
                    self.fake_keyboard_pub_.publish(self.keyboardDict["KEY_S_"])
                    print("printing current_config: {}".format(self.current_config))
                    current_config_state = self.current_config
                    self.init_config_pub_.publish(current_config_state)
                    init_config_state = self.prepare_init_goal_state(current_config_state)
                    # Now init_config_state has contact point info.
                    self.g = self.sample_goal(init_config_state)
                    goal_img_msg = self.list2imgmsg(self.g)
                    self.goal_config_pub_.publish(goal_img_msg)
                    self.initialized = True
            if self.enable and self.initialized:
                # I think I can just add that config
                # I don't need to add the self.current action stuff; I just need to collect marker position and contact positions
                # What's inside self.current_config?
                # Preprocess self.current_config
                #printing current_config: ids: [0, 1, 2, 3, 4, 5, 6, 7]
                #posx: [530, 496, 743, 779, 621, 521, 754, 0]
                #posy: [158, 265, 161, 275, 189, 494, 508, 0]
                #angles: [2.0344439357957027, 1.501939837493852, 1.174714885230589, 1.6553400823078948, 1.760276484676044, 1.6199370612109167, 1.6525807074128926, 0.0]
                # I should look at Krishnan's script to see how he recorded the keyboard control action
                print("Waiting for the action from keyboard....")
                u = ACT_CMDS_DIC[self.keyboard_input]
                if u == 8:
                    print("action command: {}".format(u))
                    continue
                else:
                    print("action command: {}".format(u))
                    acts.append(np.expand_dims([u], 0))
                    print("contact_point object left: {}".format(self.contact_point_object_left))
                    print("contact_point object right: {}".format(self.contact_point_object_right))
                    current_state = [self.current_config.posx[0], self.current_config.posy[0], self.current_config.posx[2], self.current_config.posy[2],
                                   self.current_config.posx[4], self.current_config.posy[4],
                                   self.current_config.angles[0], self.current_config.angles[2], self.current_config.angles[4],
                                   self.contact_point_object_left[0], self.contact_point_object_left[1],
                                   self.contact_point_object_right[0], self.contact_point_object_right[1]]
                    obs.append(np.expand_dims(np.array(current_state), 0))
                if self.is_dropped_ or self.is_stuck_ or self.reset_to_save_motors == True:
                    drops.append(np.expand_dims(np.array([int(self.is_dropped_)]), 0))
                    stucks.append(np.expand_dims(np.array([int(self.is_stuck_ or self.reset_to_save_motors)]), 0))

                    if self.is_dropped_:
                        print("Dropped Object")
                        rospy.loginfo("Dropped Object")
                    if self.is_stuck_ or self.reset_to_save_motors == True:
                        print("Got stuck!!!!")
                        rospy.loginfo("Stuck Object")
                        self.reset_to_save_motors == False

                    self.resetObject()
                    # prepare episode
                    print("Returning an episode....")
                    episode['o'], episode['u'], episode['drop'], episode['stuck'] = obs, acts, drops, stucks
                    return convert_episode_to_batch_major(episode)
                else:
                    drops.append(np.expand_dims(np.array([int(self.is_dropped_)]), 0))
                    stucks.append(np.expand_dims(np.array([int(self.is_stuck_ or self.reset_to_save_motors)]), 0))

            rate.sleep()


    def keyboard_callback(self, data):
        # print("keyboard data : {}".format(data.data))
        # TODO: Convert data 115 to str (such as stop, left etc)
        # (Find where the mapping is)
        if data.data == 113:
            self.keyboard_input = 'left up'
        elif data.data == 119:
            self.keyboard_input = 'up'
        elif data.data == 101:
            self.keyboard_input = 'right up'
        elif data.data == 97:
            self.keyboard_input = 'left'
        elif data.data == 115:
            self.keyboard_input = 'stop'
        elif data.data == 100:
            self.keyboard_input = 'right'
        elif data.data == 122:
            self.keyboard_input = 'left down'
        elif data.data == 120:
            self.keyboard_input = 'down'
        elif data.data == 99:
            self.keyboard_input = 'right down'
        else:
            print('Error: Keyboard input is invalid: {}'.format(data.data))


    def generate_rollouts(self):
        """Performs 'rollout_batch_size' rollouts in parallel for time horizon 'T' with the current policy
         acting on it accordingly.
         **NOTE** For ModelT42, rollout_batch_size is 1. For simulation, it could be larger than 1.
         """
        #self.reset_all_rollouts()
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
        achieved_goals, goals, stucks = [], [], []

        episode = dict(o=None, u=None, r=None, done=None, mu=None, ag=None, drop=None, g=None, stuck=None)
        Qs = []
        self.count_random_action_sent = 0
        while not rospy.is_shutdown():
            # Instead of doing this, I want to get the system_state from ModelT42
            reset_flag = rospy.get_param('/RLAgent/reset')
            #self.initObjectPosition()
            # Let the hand move
            print("reset_flag: {}".format(reset_flag))
            print("system_state: {}, self.enable: {}, self.initialized: {}, self.count_random_action_sent: {}".format(self.system_state, self.enable, self.initialized, self.count_random_action_sent))
            if self.count_random_action_sent < 5 and self.system_state==2:
                # self.initObjectPosition()
                if self.count_random_action_sent == 1:
                    self.g = self.current_config
                self.init_object_position()
                if self.count_random_action_sent == 5:
                    print("Initialization Done")
                    self.fake_keyboard_pub_.publish(self.keyboardDict["KEY_S_"])
                    # initial_config = PointArray()
                    # initial_config.x = []
                    print("printing current_config: {}".format(self.current_config))
                    current_config_state = self.current_config
                    self.init_config_pub_.publish(current_config_state)
                    init_config_state = self.prepare_init_goal_state(current_config_state)
                    # Now init_config_state has contact point info.
                    self.g = self.sample_goal(init_config_state)
                    goal_img_msg = self.list2imgmsg(self.g)
                    self.goal_config_pub_.publish(goal_img_msg)
                    self.initialized = True
            # if reset_flag:
            if self.enable and self.initialized:
                print("self.current_action: {}".format(self.current_action))
                if True: #self.current_action is not None:  #
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
                    acts.append(np.expand_dims([self.current_action], 0))
                    mus.append(self.mus)
                    # rewards.append(np.expand_dims(np.array([self.current_reward]), 0))
                    dones.append(np.expand_dims(np.array([self.current_done]), 0))

                    if self.is_dropped_ or self.is_stuck_ or self.reset_to_save_motors == True:
                        # resp = record_hand_srv_(DROP_CASE)
                        drops.append(np.expand_dims(np.array([int(self.is_dropped_)]), 0))
                        stucks.append(np.expand_dims(np.array([int(self.is_stuck_ or self.reset_to_save_motors)]), 0))
                        rewards.append(np.expand_dims(np.array([self.current_reward-20]), 0))
                        if self.is_dropped_:
                            print("Dropped Object")
                            rospy.loginfo("Dropped Object")
                        if self.is_stuck_ or self.reset_to_save_motors == True:
                            print("Got stuck!!!!")
                            rospy.loginfo("Stuck Object")
                            self.reset_to_save_motors == False

                        self.resetObject()
                        # TODO: Check how I should return episode here.
                        # I think I can just return episode right here without caring about the system_state
                        # because this generate_rollout function is from ILRL_acer.py
                        # self.num_dropped_ = self.num_dropped_ + 1
                        print("Returning an episode....")
                        # TODO: I should fill out the np array thing if the length of the array is not full.
                        episode['o'], episode['u'], episode['r'], episode['done'] = obs, acts, rewards, dones
                        episode['mu'], episode['ag'], episode['drop'], episode['g'], episode['stuck'] = mus, achieved_goals, drops, goals, stucks

                        if not self.is_episode_shape_ok(episode):
                            episode = self.fill_episode_with_zeros(episode)
                        return convert_episode_to_batch_major(episode)
                    else:
                        drops.append(np.expand_dims(np.array([int(self.is_dropped_)]), 0))
                        stucks.append(np.expand_dims(np.array([int(self.is_stuck_ or self.reset_to_save_motors)]), 0))
                        rewards.append(np.expand_dims(np.array([self.current_reward]), 0))

                    if len(acts) == self.num_rollouts:
                        self.resetObject()
                        print("Returning an episode....")
                        # TODO: I should fill out the np array thing if the length of the array is not full.
                        episode['o'], episode['u'], episode['r'], episode['done'] = obs, acts, rewards, dones
                        episode['mu'], episode['ag'], episode['drop'], episode['g'], episode['stuck'] = mus, achieved_goals, drops, goals, stucks
                        return convert_episode_to_batch_major(episode)

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
                episode['mu'], episode['ag'], episode['drop'], episode['g'], episode['stuck'] = mus, achieved_goals, drops, goals, stucks
                if self.compute_Q:
                    return convert_episode_to_batch_major(episode)  #, np.mean(Qs)
                else:
                    return convert_episode_to_batch_major(episode)

            rate.sleep()
        #rospy.spin()

    def is_episode_shape_ok(self, episode):
        # Checking if the number of rollouts is enough for a valid episode
        # i.e. if drop/stuck happens, the rollout ends there so we have to fill up the episode with zeros
        # to make sure every episode has the same length
        print("len(episode['o']).shape: {}".format(len(episode['o'])))
        if len(episode['o']) == self.num_rollouts: #self.model.nbatch+1:
            return True
        else:
            return False

    def fill_episode_with_zeros(self, episode):
        #TODO: Finish this function
        for key, val in episode.items():
            while len(episode[key]) < self.num_rollouts: #self.model.nbatch + 1 -> self.model.nbatch
                temp_shape = episode[key][0].shape
                print("temp_shape: {}".format(temp_shape))
                episode[key].append(np.zeros(temp_shape))
            print("len(episode[key]): {}".format(len(episode[key])))
        # all the length of episodes are self.model.nbatch + 1
        return episode

    def itemDroppedCallback(self, msg):
        self.is_dropped_ = msg.data

    def itemStuckCallback(self, msg):
        self.is_stuck_ = msg.data

    def objectMoveDist(self, msg):
        self.object_move_dist_ = msg.data

    def gripperLoadCallback(self, msg):  # We will only keep track of when load history is exceeded
        #print("gripperLoad: {}")
        history = msg.data
        if len(history) > 0:
            # Do the two motor case first
            if len(self.load_history_) >= self.load_history_length_:
                self.load_history_.pop(0)  # Remove the first one

            if history[0] < (-800) and history[1] < (-800):
                self.load_history_.append(1)
            else:
                self.load_history_.append(0)

            # Now do the two motor case
            if len(self.single_motor_save_) >= self.load_history_length_ + 20:
                self.single_motor_save_.pop(0)  # Remove the first

            temp0 = history[0] < (-800)
            temp1 = history[1] < (-800)
            # print("gripperLoad: {}, {}".format(history[0], history[1]))
            if abs(history[0]) > 750 or abs(history[1]) > 750:
                print("ggg load is too much: {}, {}".format(history[0], history[1]))
                self.reset_to_save_motors = True
            else:
                self.reset_to_save_motors = False

    def resetObject(self):
        print("resetting object!!!!!!!!!!!!!!!!!!!!!1")
        time_to_break = False
        self.enable_fake_keyboard_= False
        self.enable = False
        fake_keyboard_pub_.publish(self.keyboardDict["KEY_S_"])
        initialize_hand_srv_(OPEN_GRIPPER) # let the system stop
        time.sleep(8)

        counter = 0
        while time_to_break == False:
            resp =  reset_obj_srv_(RAISE_CRANE_RESET)
            time.sleep(5)
            tic = time.time()
            print("self.object_move_dist: {}".format(self.object_move_dist_))
            while self.object_move_dist_ >15:
                toc = time.time()
                time.sleep(0.5)
                if (toc - tic > seconds_until_next_reset):
                    break
            if self.object_move_dist_ <=15:
                time_to_break=True
            counter += 1
            if counter == 3:
                break

        #Now we can close the fingers and wait for the grasp
        print("Getting gripper ready")
        initialize_hand_srv_(GET_GRIPPER_READY)
        time.sleep(8)
        print("lowering crane object ready")
        reset_obj_srv_(LOWER_CRANE_OBJECT_READY)
        time.sleep(1.5)
        #time.sleep(10)
        print("allowing gripper to move")
        initialize_hand_srv_(ALLOW_GRIPPER_TO_MOVE)
        self.enable_fake_keyboard_ = True
        self.enable = True
        self.is_dropped_= False
        self.is_stuck_ = False
        self.count_random_action_sent = 0
        self.initialized = False
        self.master_tic = time.time()