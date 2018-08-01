import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

import sys
sys.path.append("/home/grablab/grablab-ros/src/projects/sliding_policies")
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

'''
data_dict = {"train_x": train_x, "test_x": test_x, "train_y": train_y, "test_y": test_y}
with open('data_dict.pkl', 'wb') as f:
    pickle.dump(data_dict, f)
'''

'''
dataset = DataSet(**{'_data':train_x, '_labels':train_y, '_test_data': test_x, '_test_labels': test_y})

import pickle
with open('dataset_train_ilpolicy.pkl', 'wb') as f:
    pickle.dump(dataset, f)
'''