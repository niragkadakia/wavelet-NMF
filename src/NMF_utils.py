"""
Utilities for NMF factorization.

Created by Adam Fine and Nirag Kadakia at 11:11 09-12-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


def compute_num_factors(model):
	"""
	Get number of nonzero W factors.
	"""
	
	factor = 0
	for factor in range(model.W.shape[2]):
		if not model.W[:, :, factor].any():
			return factor
	return model.W.shape[2]
