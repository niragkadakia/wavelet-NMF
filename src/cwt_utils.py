"""
Utilities for continuous wavelet transform

Created by Adam Fine and Nirag Kadakia at 15:30 08-21-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import pywt

def get_cwt_scales(wavelet, lower_freq, upper_freq, dt, num):
	"""
	Get expected range of frequencies for cwt.
	"""
	
	central_freq = pywt.central_frequency(wavelet, precision=15)
	scales = np.geomspace(central_freq/(upper_freq*dt), 
			   central_freq/(lower_freq*dt), num=num, endpoint=True)
	
	return scales