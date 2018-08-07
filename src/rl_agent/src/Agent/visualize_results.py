import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.switch_backend('GtkAgg')
import numpy as np

q_val = np.load('./results/q_val.npy')
c_loss = np.load('./results/cri_loss.npy')
a_loss = np.load('./results/actor_loss.npy')
bc_loss = np.load('./results/bc_loss.npy')
#ent = np.load('./results/ent.npy')

#plt.scatter(np.arange(len(q_val)), q_val, c='b', label='q-value')
#plt.scatter(np.arange(len(bc_loss)), bc_loss, c='green', label='bc loss')
#plt.scatter(np.arange(len(ent)), ent, c='purple', label='ent')
plt.scatter(np.arange(len(c_loss)), c_loss, c='yellow', label='critic loss')
#plt.scatter(np.arange(len(a_loss)), a_loss, c='red', label='actor loss')
#plt.title('critic loss')
plt.legend()
plt.show()
