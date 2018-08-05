#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

import rospy
import math
import datetime

from rl_msgs.msg import RLStateReward, RLAction
from std_msgs.msg import Float32MultiArray

from policies import MlpPolicy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

import sys
sys.path.append("/home/grablab/grablab-ros/src/projects/sliding_policies")
from dataset import DataSet

ACT_CMDS = ['up', 'down', 'left', 'right', 'left up', 'left down', 'right up', 'right down', 'stop']
MODEL_SAVE_PATH = os.path.join(os.environ['HOME'], 'grablab-ros/src/external/rl-texplore-ros-pkg/src/rl_agent/src/Agent/')

def get_data():
    file_date = "Jun2714152018_eps1"
    file_date2 = "Jun2714312018_eps1"
    file_date3 = "Jul816002018_eps1"
    file_date4 = "Jul816002018_eps3"


    data_file = "~/grablab-ros/src/projects/sliding_policies/data/yutaro_test_bag_" + file_date +".csv"
    data_file2 = "~/grablab-ros/src/projects/sliding_policies/data/yutaro_test_bag_" + file_date2 +".csv"
    data_file3 = "~/grablab-ros/src/projects/sliding_policies/data/yutaro_test_bag_" + file_date3 +".csv"
    data_file4 = "~/grablab-ros/src/projects/sliding_policies/data/yutaro_test_bag_" + file_date4 +".csv"

    file_date = file_date + "_" + file_date2 + "_" + file_date3 + "_" + file_date4

    df = pd.read_csv(data_file, header=0)
    df2 = pd.read_csv(data_file2, header=0)
    df3 = pd.read_csv(data_file3, header=0)
    df4 = pd.read_csv(data_file4, header=0)

    df = df[df.last_action_keyboard != 'stop']
    df2 = df2[df2.last_action_keyboard != 'stop']
    df3 = df3[df3.last_action_keyboard != 'stop']
    df4 = df4[df4.last_action_keyboard != 'stop']

    df = df.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
    df4 = df4.dropna()

    ACT_FNS = {'relu': tf.nn.relu}
    SAVE_PATH = os.path.join(os.environ['HOME'], 'grablab-ros/src/projects/sliding_policies/models/il_actor')


    def prepare_data(df, shift=True):
        if shift:
            labels = df['last_action_keyboard'][3:]
            data = df.iloc[:-3, :9].values
            print("data shape: {}".format(data.shape))
        else:
            labels = df['last_action_keyboard']
            data = df.iloc[:, :9].values
            print("data shape: {}".format(data.shape))

        # scale the first 6 dim (corresponding to the 3 markers x,y info) so that marker pos and marker angle have
        # the same scale
        # data[:,:6] /= 100.0 # this is determined by looking at the summary stats from df

        print('data sclaing sanity check...')
        for i in range(9):
            print(np.max(data[:, i]))

        ACT_CMDS = ['up', 'down', 'left', 'right', 'left up', 'left down', 'right up', 'right down', 'stop']
        bin_labels = label_binarize(labels, ACT_CMDS)  # bin_labels = one-hot vectors

        le = LabelEncoder()
        enc_labels = le.fit_transform(labels)

        print("bin_labels: {}".format(bin_labels[:5, :]))
        print("enc_labels: {}".format(enc_labels[:5]))
        print("# enc_labels are passed to stratify parameter in train_test_split. ")

        return data, bin_labels, enc_labels
        # train_x, test_x, train_y, test_y = train_test_split(data, bin_labels, stratify=enc_labels, train_size=0.8)
        # return DataSet(**{'_data':train_x, '_labels':train_y, '_test_data': test_x, '_test_labels': test_y})


    df_data, df_bin_labels, df_enc_labels = prepare_data(df)
    df2_data, df2_bin_labels, df2_enc_labels = prepare_data(df2)
    df3_data, df3_bin_labels, df3_enc_labels = prepare_data(df3)
    df4_data, df4_bin_labels, df4_enc_labels = prepare_data(df4)


    data = np.concatenate([df_data, df2_data], axis=0)
    bin_labels = np.concatenate([df_bin_labels, df2_bin_labels], axis=0)
    enc_labels = np.concatenate([df_enc_labels, df2_enc_labels])
    print(data.shape)

    data = np.concatenate([data, df3_data], axis=0)
    bin_labels = np.concatenate([bin_labels, df3_bin_labels], axis=0)
    enc_labels = np.concatenate([enc_labels, df3_enc_labels])
    print(data.shape)

    data = np.concatenate([data, df4_data], axis=0)
    bin_labels = np.concatenate([bin_labels, df4_bin_labels], axis=0)
    enc_labels = np.concatenate([enc_labels, df4_enc_labels])
    print(data.shape)

    train_x, test_x, train_y, test_y = train_test_split(data, bin_labels, stratify=enc_labels, train_size=0.8)

    np.save("train_x.npy", train_x)
    np.save("train_y.npy", train_y)
    np.save("test_x.npy", test_x)
    np.save("test_y.npy", test_y)


class ILRLAgent(object):
    """
    Loads a TensorFlow model and turns it into an agent that publishes to rl_agent/rl_action
    """

    def __init__(self, agent):
        super(ILRLAgent, self).__init__()
        self.agent = agent #ILPolicy(num_states=num_states, num_actions=num_actions, model_name=model_name)
        #self.agent.build()  # creates computation graph for neural network
        #self.agent.load_model()  # assumes model is trained already
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
            # print("a0,a2,a4 in prepare_s: {}, {}, {}".format(a0, a2, a4))
            # x0 /= 100; y0 /= 100; x2 /= 100; y2 /= 100; x4 /= 100; y4 /= 100
            # print("x0: {}".format(x0))
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

    #def eval_agent(self, s):
    #    return self.agent.eval([s])[0]

    def next_action(self, s, r):
        self.rewards.append(r)
        self.actions.append(self.current_action)
        self.states.append(self.current_state)
        x4, y4 = s[-2:] if len(s) == 6 else s[-5:-3]
        dist_to_goal = math.sqrt((x4 - self.goal[0]) ** 2 + (y4 - self.goal[1]) ** 2)
        # if dist_to_goal < self.goal_range:  # if we are within range of goal send stop command (?)
        #    return 8
        # print("state in next_action func: {}".format(s))
        sampled_ac = self.agent.get_actions(s)  #self.eval_agent(s)
        self.current_action, self.current_state = sampled_ac, s
        return sampled_ac



class IL(object):
    def __init__(self, policy, num_states, num_actions, nsteps, nenvs=1, batch_size=40,
                 save_path=None, model_name=None, restore=False, checkpoint_path=None):
        self.y_ = tf.placeholder(tf.float32, [None, num_actions])
        self.model_name = model_name
        self.save_path = save_path
        self.num_states = num_states; self.num_actions = num_actions

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        train_model = policy(self.sess, num_states, num_actions, nenvs * nsteps, nsteps, reuse=False, deterministic=True)

        # params = self._find_trainable_variables("model/pi")
        params = self._vars("model/pi")
        for i in params:
            print(i)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                    labels=self.y_,
                                    logits=train_model.pi))
        grads = tf.gradients(self.loss, params)
        #if max_grad_norm is not None:
        #    grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer()
        self.opt_op = trainer.apply_gradients(grads)

        self.saver = tf.train.Saver()

        # restore if pretrained = True
        if restore:
            pi_var_list = self._vars("model/pi")
            self._restore(pi_var_list, checkpoint_path)

        self.train_model = train_model
        tf.global_variables_initializer().run(session=self.sess)

    def _restore(self, pi_var_list, checkpoint_path):
        saver = tf.train.Saver(var_list=pi_var_list)
        saver.restore(self.sess, checkpoint_path)

    '''
    def _find_trainable_variables(self, key):
        with tf.variable_scope(key):
            return tf.trainable_variables()
    '''

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        assert len(res) > 0
        return res

    def step(self, batch_x, batch_y):
        loss, _ = self.sess.run([self.loss, self.opt_op], feed_dict={self.train_model.X: batch_x, self.y_: batch_y})
        return loss

    def score(self, x, y):
        action = self.sess.run(self.train_model.pi, feed_dict={self.train_model.X : x})
        y_pred = np.argmax(action, axis=-1)
        true_classes = np.argmax(y, axis=1)
        acc = np.sum(true_classes == y_pred) / float(len(y_pred))
        return acc

    def save_model(self, model_name=None):
        model_name = model_name or self.model_name
        return self.saver.save(self.sess, os.path.join(self.save_path, model_name))

    def get_actions(self, o, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        act = self.train_model.step(o)
        print(act)
        return act[0]


def train():
    policy = MlpPolicy
    dims = {'o': 9, 'u': 9}
    nsteps = 1
    model_name = 'il_policy_for_a2c'
    MODEL_SAVE_PATH = './'
    model = IL(policy, num_states=dims['o'], num_actions=dims['u'], nsteps=nsteps,
                  model_name=model_name, save_path=MODEL_SAVE_PATH)

    #import sys; sys.exit()

    losses, accuracies, train_accs = [], [], []
    best_acc = .50

    num_steps = 30000 #30000 # 100000
    batch_size = 100

    train_x = np.load('train_x.npy')
    train_y = np.load('train_y.npy')
    test_x = np.load('test_x.npy')
    test_y = np.load('test_y.npy')
    dataset = DataSet(**{'_data': train_x, '_labels':train_y, '_test_data':test_x, '_test_labels':test_y})

    for step in range(num_steps):
        batch_x, batch_y = dataset.next_batch(batch_size)
        loss = model.step(batch_x, batch_y)
        train_acc = model.score(batch_x, batch_y)
        train_accs.append(train_acc)
        if (step + 1) % 500 == 0:
            acc = model.score(dataset.test_data, dataset.test_labels)
            accuracies.append(acc)
            if acc > best_acc:
                best_acc = acc
                save_loc = model.save_model()
                print('saved model at: {} after {} steps'.format(save_loc, step + 1))
    losses.append(loss)

    import matplotlib.pyplot as plt

    plt.plot(train_accs)
    plt.show()

    plt.plot(accuracies)
    plt.title('max test acc: {:3.2f}%'.format(100 * max(accuracies)))
    plt.show()


def evaluate():
    policy = MlpPolicy
    dims = {'o': 9, 'u': 9}
    nsteps = 1
    model_name = 'il_policy_for_a2c'
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, model_name)
    model = IL(policy, num_states=dims['o'], num_actions=dims['u'], nsteps=nsteps,
                  model_name=model_name, checkpoint_path=checkpoint_path, restore=True)

    node_name = 'RLAgent'
    rollout(model, node_name)


def rollout(model, node_name):
    rospy.init_node(node_name)
    rospy.loginfo('started RLAgent node')
    rate = rospy.Rate(5)
    agent = ILRLAgent(model)

    while not rospy.is_shutdown():
        reset_flag = rospy.get_param('/RLAgent/reset')
        if reset_flag:
            if agent.current_action is not None:  #
                rospy.loginfo('sampled action: {}'.format(ACT_CMDS[agent.current_action]))
                rla = RLAction(action=agent.current_action)
                #print("RLAction msg has been published from ImitationLearning.py!!!")
                agent.ac_pub.publish(rla)
        else:
            if len(agent.states) > 0:
                #agent.log_data()
                agent.reset()
        rate.sleep()

if __name__ == '__main__':
    #train()
    evaluate()
