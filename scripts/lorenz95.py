"""
Generate Lorenz96 chaotic model dynamics. Use this to get wavelet transform
and pick out spike events using CoNMF technique

Created by Nirag Kadakia at 20:30 09-19-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lorenz96(x, t):
	return (np.roll(x, -1) - np.roll(x, 2))*np.roll(x, 1) - x + 8

t = np.linspace(0, 50, 1000)
x0 = np.random.uniform(-10, 10, 5)

sol = odeint(lorenz96, x0, t)

dir = 'C:/Users/nk479/Dropbox (emonetlab)/users' +\
		'/nirag_kadakia/data/beh-NMF/lorenz95'
data = np.vstack((t, sol.T)).T
np.savetxt('%s/0.txt' % dir, data, fmt='%.3f')
plt.plot(sol)
plt.show()