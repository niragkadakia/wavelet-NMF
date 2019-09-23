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

def lorenz63(X, t, s=10, r=28, b=2.667):
	x, y, z = X
	x_dot = s*(y - x)
	y_dot = r*x - y - x*z
	z_dot = x*y - b*z
	return x_dot, y_dot, z_dot
	
t = np.linspace(0, 50, 5000)
x0 = [0, 1.5, 1.05]

sol = odeint(lorenz63, x0, t)

dir = 'C:/Users/nk479/Dropbox (emonetlab)/users' +\
		'/nirag_kadakia/data/beh-NMF/lorenz63'
data = np.vstack((t, sol.T)).T
np.savetxt('%s/0.txt' % dir, data, fmt='%.3f')
plt.plot(sol)
plt.show()

plt.plot(sol[:, 0], sol[:, 1])
plt.show()