import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.switch_backend('GtkAgg')
import numpy as np

q_val = np.load('q_val.npy')

plt.scatter(np.arange(len(q_val)), q_val)
plt.show()
