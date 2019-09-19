import numpy as np 
import matplotlib.pyplot as plt

import sys
sys.path.append("../src")
sys.path.append("../src/cnmfpy")
from beh_NMF import *
from scipy.stats import special_ortho_group

a = wavelet_transform(exp_dir='lorenz95', exp_name='0')

raw_data = load_raw_data(a.exp_dir, a.exp_name)

beg = 0
xy = raw_data[beg:, 1:]
num_vars = xy.shape[1]
print (num_vars)

#plt.plot(xy[:, 0], xy[:, 1])
#plt.show()

data = None
for i in range(50):
	R = special_ortho_group.rvs(num_vars)
	rot_xy = np.dot(R, xy.T)
	
	if data is None:
		data = abs(rot_xy[0])
	else:
		data = np.vstack((data, abs(rot_xy[0])))

pow = data[:, 0]#np.sum(data, axis=1)
idxs = np.argsort(pow)

plt.subplot(311)
plt.plot(xy[:, 0])
plt.xlim(beg, data.shape[-1])
plt.subplot(312)
plt.plot(xy[:, 1])
plt.xlim(beg, data.shape[-1])
plt.subplot(313)
plt.pcolormesh(data[idxs])
plt.xlim(beg, data.shape[-1])
plt.show()
