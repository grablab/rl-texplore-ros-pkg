#!/usr/bin/python
from __future__ import print_function

import rospy
import tensorflow as tf
import numpy as np
import os
import math
import datetime

from rl_msgs.msg import RLStateReward, RLAction
from std_msgs.msg import Float32MultiArray

from NeuralQFittedIteration import NFQ
import keras

ACT_FNS = {'relu': tf.nn.relu}
NODE = "RLAgent"
MODE = 'il' # nfq
MODEL_SAVE_PATH = os.path.join(os.environ['HOME'], 'grablab-ros/src/projects/sliding_policies/models/'+MODE+'/')
DATA_SAVE_PATH = os.path.join(os.environ['HOME'], 'grablab-ros/src/projects/sliding_policies/data/')
ACT_CMDS = ['up', 'down', 'left', 'right', 'left up', 'left down', 'right up', 'right down', 'stop']
POL_TO_ACT_CMDS = {0: 1, 1: 2, 2: 3, 3: 8, 4: 0}
POL_TO_ACT_CMDS2 = {0:0, 1:1, 2:2, 3:4, 4:5}


class ILPolicy:
    def __init__(self, num_states=14, num_actions=9, n_layers=4, hidden_units=[100, 60, 40, 20], act_fn='relu', model_name='il_policy',
                 save_path=MODEL_SAVE_PATH):
        self.num_states = num_states
        self.num_actions = num_actions
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.act_fn = ACT_FNS.get(act_fn, None)
        self.model_name = model_name
        self.save_path = save_path
        self.sess, self.saver = None, None

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

        self.sess.run(tf.global_variables_initializer())

    def load_model(self, model_name=None):
        model_name = model_name or self.model_name
        self.saver.restore(self.sess, os.path.join(self.save_path, model_name))
        print('model restored')

    def save_model(self, model_name=None):
        """returns save path"""
        model_name = model_name or self.model_name
        return self.saver.save(self.sess, os.path.join(self.save_path, model_name))

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

# def process_env_description(env_desc):
#     num_actions = env_desc.num_actions
#     num_states = env_desc.num_states
#     agent = ILRLAgent(num_actions, num_states)
#     return agent

'''
class RLBase(object):
    def __init__(self):
        print('RLBase created')
        rospy.Subscriber('/gripper/loa', Float32MultiArray, self.gripper_load_callback)

    def gripper_load_callback(self, data):
        print("gripper_load_callback data : {}".format(data))
'''

class ILRLAgent(object):
    """
    Loads a TensorFlow model and turns it into an agent that publishes to rl_agent/rl_action
    """
    def __init__(self, num_states, num_actions, model_name='il_policy'):
        super(ILRLAgent, self).__init__()
        self.agent = ILPolicy(num_states=num_states, num_actions=num_actions, model_name=model_name)
        self.agent.build()  # creates computation graph for neural network
        self.agent.load_model()  # assumes model is trained already 
        self.init_ros_nodes()
    
    def init_ros_nodes(self):
        self.goal = [int(x) for x in rospy.get_param('/RLAgent/goal').split(',')]
        self.goal_range = rospy.get_param('/RLAgent/goal_range')

        self.current_state = None
        self.current_action = None
        self.states, self.actions, self.rewards = [], [], []
        self.terminated = False
        rospy.Subscriber('rl_env/rl_state_reward', RLStateReward, self.state_reward_callback)
        self.ac_pub = rospy.Publisher('rl_agent/rl_action', RLAction, queue_size=1)

        self.load = [0]
        rospy.Subscriber('gripper/load', Float32MultiArray, self.load_callback)
        rospy.Subscriber('gripper/curr', Float32MultiArray, self.curr_callback)

    def reset(self):
        self.current_action = None
        self.current_state = None
        self.terminated = False
        self.states, self.actions, self.rewards = [], [], []

    def log_data(self, logfile=None):
        logfile = logfile or datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + '.csv'
        logfile = os.path.join(DATA_SAVE_PATH, logfile)
        with open(logfile, 'w') as f:
            f.write('state, action, reward')
            for s, a, r in zip(self.states, self.actions, self.rewards):
                f.write('\n{},{},{}'.format(s, a, r))

    def state_reward_callback(self, sr):
        if not self.terminated:
            if sr.terminal:
                self.terminated = True
                return
            s = self.prepare_s(sr.state)
            if self.current_action is None:
                ac = self.first_action(s)
            else:
                ac = self.next_action(s, sr.reward)
        else:
            return
        # rospy.loginfo("state: {}, action: {}".format(sr.state, ac))

    def load_callback(self, data):
        self.load = data.data

    def curr_callback(self, data):
        self.curr = data.data

    def prepare_s(self, s):
        if len(s) == 17:
            x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, a0, a2, a4 = s
        elif len(s) == 9:
            x0, y0, x2, y2, x4, y4, a0, a2, a4 = s
        if self.agent.num_states == 6:
            s = [x0, y0, x2, y2, x4, y4]
        elif self.agent.num_states == 9:
            #print("a0,a2,a4 in prepare_s: {}, {}, {}".format(a0, a2, a4))
            #x0 /= 100; y0 /= 100; x2 /= 100; y2 /= 100; x4 /= 100; y4 /= 100
            #print("x0: {}".format(x0))
            s = [x0, y0, x2, y2, x4, y4, a0, a2, a4]
        else:
            assert False, 'need to set num_states to 6 or 9'
        return s

    def first_action(self, s):
        self.current_action = 8
        self.current_state = s
        self.states.append(s)
        self.actions.append(self.current_action)
        self.rewards.append(0.)
        return self.current_action

    def eval_agent(self, s):
        return self.agent.eval([s])[0]

    def next_action(self, s, r):
        self.rewards.append(r)
        self.actions.append(self.current_action)
        self.states.append(self.current_state)
        x4, y4 = s[-2:] if len(s) == 6 else s[-5:-3]
        dist_to_goal = math.sqrt((x4-self.goal[0])**2 + (y4-self.goal[1])**2)
        # if dist_to_goal < self.goal_range:  # if we are within range of goal send stop command (?)
        #    return 8
        #print("state in next_action func: {}".format(s))
        sampled_ac = self.eval_agent(s)
        if self.agent.num_actions == 5:  # this is for krishnan's ILPolicy
            sampled_ac = POL_TO_ACT_CMDS2[sampled_ac]
        self.current_action, self.current_state = sampled_ac, s
        return sampled_ac


class NFQRLAgent(ILRLAgent):
    def __init__(self, num_states, num_actions, model_name='nfq_weights.best.hdf5'):
        model_path = os.path.join(MODEL_SAVE_PATH, model_name)
        # This is where the graph is built. Which thread is this using?
        self.agent = NFQ(num_states, num_actions, terminal_states=None)
        if model_name and os.path.exists(model_path):
            self.agent.load_model(model_path)
        self.init_ros_nodes()

    def eval_agent(self, s):
        print("State in def eval_agent func: {}".format(np.asarray(s)))
        return self.agent.greedy_action(np.asarray(s))

if __name__ == '__main__':

    rospy.init_node(NODE)
    rospy.loginfo('started RLAgent node')
    policy_type = 'Jun2714152018_eps1_Jun2714312018_eps1_Jul816002018_eps1' #'Jun2714152018_eps1' #'Jul816002018_eps1' #'yutaro' #'
    # policy_type = 'Jun2714152018_eps1_Jun2714312018_eps1_Jul816002018_eps1_Jul816002018_eps3' #'Jun2714152018_eps1' #'Jul816002018_eps1' #'yutaro' #'
    if policy_type == 'krishnan':
        NUM_STATES = 6
        NUM_ACTIONS = 5
        MODEL_NAME = 'il_pol4'
    elif policy_type == 'yutaro':
        NUM_STATES, NUM_ACTIONS, MODEL_NAME = 6, 9, 'il_pol_yutaro'
    elif policy_type == 'Jun2714152018_eps1_Jun2714312018_eps1_Jul816002018_eps1':
        NUM_STATES, NUM_ACTIONS, MODEL_NAME = 9, 9, 'il_pol_' + policy_type
    rl_method = 'il' #'nfq'
    if rl_method == 'il':
        agent = ILRLAgent(NUM_STATES, NUM_ACTIONS, MODEL_NAME)
    elif rl_method == 'nfq':
        NUM_STATES, NUM_ACTIONS = 9, 9
        MODEL_NAME = 'nfq_weights.best.hdf5'
        agent = NFQRLAgent(NUM_STATES, NUM_ACTIONS, MODEL_NAME)
    rate = rospy.Rate(5)  # publishes action every .2 seconds

    while not rospy.is_shutdown():
        reset_flag = rospy.get_param('/RLAgent/reset')
        if reset_flag:
            if agent.current_action is not None:  #
                rospy.loginfo('RLAgent load: {}, curr: {}, sampled action: {}'.format(
                        agent.load, agent.curr, ACT_CMDS[agent.current_action]))
                rla = RLAction(action=agent.current_action)
                print("RLAction msg has been published from ImitationLearning.py!!!")
                agent.ac_pub.publish(rla)
        else:
            if len(agent.states) > 0:
                agent.log_data()
                agent.reset()
        rate.sleep()




# if __name__ == '__main__':
#
#     rospy.init_node(NODE)
#     rospy.loginfo('started RLAgent node')
#     policy_type = 'Jun2714152018_eps1_Jun2714312018_eps1_Jul816002018_eps1' #'Jun2714152018_eps1' #'Jul816002018_eps1' #'yutaro' #'
#     # policy_type = 'Jun2714152018_eps1_Jun2714312018_eps1_Jul816002018_eps1_Jul816002018_eps3' #'Jun2714152018_eps1' #'Jul816002018_eps1' #'yutaro' #'
#     if policy_type == 'krishnan':
#         NUM_STATES = 6
#         NUM_ACTIONS = 5
#         MODEL_NAME = 'il_pol4'
#     elif policy_type == 'yutaro':
#         NUM_STATES, NUM_ACTIONS, MODEL_NAME = 6, 9, 'il_pol_yutaro'
#     elif policy_type == 'Jun2714152018_eps1_Jun2714312018_eps1_Jul816002018_eps1':
#         NUM_STATES, NUM_ACTIONS, MODEL_NAME = 9, 9, 'il_pol_' + policy_type
#     rl_method = 'il' #'nfq'
#     if rl_method == 'il':
#         agent = ILRLAgent(NUM_STATES, NUM_ACTIONS, MODEL_NAME)
#     elif rl_method == 'nfq':
#         NUM_STATES, NUM_ACTIONS = 9, 9
#         MODEL_NAME = 'nfq_weights.best.hdf5'
#         agent = NFQRLAgent(NUM_STATES, NUM_ACTIONS, MODEL_NAME)
#     rate = rospy.Rate(5)  # publishes action every .2 seconds
#     last_action, since_ac_change = -1, 0
#     last_state, since_state_change = 0, 0
#     while not rospy.is_shutdown():
#         reset_flag = rospy.get_param('/RLAgent/reset')
#         if reset_flag:
#             # if agent.load[0] < -900.0 or agent.load[1] < -900.0:
#             #     print("load is too much so I'm resetting!!!")
#             #     agent.reset()
#             if agent.current_action is not None:  #
#                 if since_ac_change > 5:
#                     agent.current_action = 8
#                 if since_ac_change < 8:
#                     rospy.loginfo('RLAgent load: {}, curr: {}, sampled action: {}, steps since change: {}'.format(
#                         agent.load, agent.curr, ACT_CMDS[agent.current_action], since_ac_change))
#                 rla = RLAction(action=agent.current_action)
#                 print("RLAction msg has been published from ImitationLearning.py!!!")
#                 agent.ac_pub.publish(rla)
#             if last_action == agent.current_action:
#                 since_ac_change += 1
#             else:
#                 since_ac_change = 0
#             if len(agent.states) == last_state:
#                 since_state_change += 1
#             else:
#                 since_state_change = 0
#                 last_state = len(agent.states)
#             last_action = agent.current_action
#             # Ask Krishnan: Why do you need this? Can't you just do all of these stuff in ModelT42.cc?
#             #if since_state_change == 75:  # 15 seconds without states published, break loop
#             #    break
#         else:
#             if len(agent.states) > 0:
#                 agent.log_data()
#                 agent.reset()
#                 since_ac_change = 0
#         rate.sleep()
