import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from dataset import DataSet

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

class ActorCritic:
