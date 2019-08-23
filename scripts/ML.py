"""
Generate Morris Lecar dynamics. Use this to get wavelet transform
and pick out spike events using CoNMF technique

Created by Nirag Kadakia at 20:30 04-22-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def I(t):
	return np.interp(t, t_vec, I_vec)
	
def Mss(V, V1, V2):
	return 0.5*(1. + np.tanh((V - V1)/V2))
	
def Nss(V, V3, V4):
	return 0.5*(1. + np.tanh((V - V3)/V4))
	
def df(x, t, C, gL, gCa, gK, tau, VL, VCa, VK, V1, V2, V3, V4):
	V, N = x
	injI = I(t)
	
	dVdt = 1./C*(injI
			   - gL*(V - VL)
			   - gCa*Mss(V, V1, V2)*(V - VCa)
			   - gK*N*(V - VK))
	dNdt = (Nss(V, V3, V4) - N)/tau
	
	return [dVdt, dNdt]
	
	
C = 20
gL = 2
gCa = 4
gK = 8
tau = 7
VL = -60
VCa = 120
VK = -84
V1 = -1.2
V2 = 18
V3 = 2
V4 = 30

for seed in range(10):
	
	# New seed; generate data
	np.random.seed(seed)
	I_vec = np.random.normal(100, 20, 1000) 
	t_vec = np.linspace(0, 10000, 1000)
	params = (C, gL, gCa, gK, tau, VL, VCa, VK, V1, V2, V3, V4)
	t = np.linspace(0, 2000, 20001)
	y0 = np.array([20., 0.5])
	sol = odeint(df, y0, t, args=params)

	# Plot
	#plt.subplot(211)
	#plt.plot(sol[:, 0])
	#plt.subplot(212)
	#plt.plot(sol[:, 1])
	#plt.show()

	# Save
	np.savetxt('../data/%s.txt' % seed, sol, fmt='%.4f', delimiter='\t')
	