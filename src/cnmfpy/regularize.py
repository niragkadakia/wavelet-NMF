import numpy as np
from numpy.linalg import norm
from scipy.signal import convolve2d
from conv import ShiftMatrix, tensor_transconv

EPSILON = np.finfo(np.float32).eps

def compute_scfo_reg(data, W, H, shifts, kernel):
	# smooth H
	maxlag = int((len(shifts) - 1) / 2)
	smooth_H = _smooth(H.shift(0).T, kernel)

	# penalize H
	pen_H = ShiftMatrix(np.dot(data.shift(0), smooth_H), maxlag)

	# penalize W
	penalty = tensor_transconv(W, pen_H, shifts)
	return norm(penalty)


def compute_scfo_gW(data, W, H, shifts, kernel):
	K, T = H.shape

	# preallocate
	scfo_gradW = np.empty(W.shape)

	# smooth H
	smooth_H = _smooth(H.shift(0).T, kernel)

	not_eye = np.ones((K, K)) - np.eye(K)

	# TODO: broadcast
	for l, t in enumerate(shifts):
		scfo_gradW[l] = data.shift(-t).dot(smooth_H).dot(not_eye)

	return scfo_gradW

"""
def compute_comb_gW(W, num_vars):
	comb_gW = np.zeros(W.shape)
	ends = np.zeros(num_vars+1, dtype=int)
	ends[0]=0
	for i in np.arange(1,num_vars+1):
		ends[i] = i*W.shape[1]//num_vars
	for pattern in np.arange(W.shape[2]):
		full_pattern_factor = 1
		for i in np.arange(num_vars):
			full_pattern_factor *= np.sum(W[:,ends[i]:ends[i+1],:])
		for i in np.arange(num_vars):
			comb_gW[:,ends[i]:ends[i+1],pattern] = \
			  full_pattern_factor/np.sum(W[:,ends[i]:ends[i+1],:])
	return np.multiply(-comb_gW, compute_comb_reg(W, num_vars)**2)

def compute_comb_reg(W, num_vars):
	ends = np.zeros(num_vars + 1, dtype=int)
	ends[0] = 0
	for i in np.arange(1, num_vars + 1):
		ends[i] = i * W.shape[1] // num_vars
	comb_denom = 0
	for pattern in np.arange(W.shape[2]):
		full_pattern_factor = 1
		for i in np.arange(num_vars):
			full_pattern_factor *= np.sum(W[:, ends[i]:ends[i+1],:])
		comb_denom += full_pattern_factor
	return 1/(comb_denom+EPSILON)
"""

def compute_scfo_gH(data, W, H, shifts, kernel):
	K, T = H.shape

	# smooth data
	maxlag = int((len(shifts) - 1) / 2)
	smooth_data = ShiftMatrix(_smooth(data.shift(0), kernel), maxlag)

	not_eye = np.ones((K, K)) - np.eye(K)

	# apply transpose convolution
	return not_eye.dot(tensor_transconv(W, smooth_data, shifts))

def compute_smooth_kernel(maxlag):
	# TODO check
	return np.ones([1, 4 * maxlag + 1])


def _smooth(X, kernel):
	return convolve2d(X, kernel, 'same')
