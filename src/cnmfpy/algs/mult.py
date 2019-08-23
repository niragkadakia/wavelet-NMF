import numpy as np
from conv import ShiftMatrix, tensor_transconv
from optimize import compute_loss, renormalize, shift_factors
from regularize import compute_scfo_gW, compute_scfo_gH, compute_scfo_reg #compute_comb_gW, compute_comb_reg 

EPSILON = np.finfo(np.float32).eps


def fit_mult(data, model):
	m, n = data.shape
	shifts = model._shifts

	# initial loss/norms
	model.loss_hist = [compute_loss(data, model.W, model.H, shifts)]
	#model.comb_norm = [compute_comb_reg(model.W, model.num_vars)]
	model.seq_norm = [compute_scfo_reg(data, model.W, model.H, shifts, model._kernel)]

	#print(model.loss_hist)

	converged, itr = False, 0
	for itr in range(model.n_iter_max):
		# shift factors
		if ((itr % 5 == 0) and (model.n_iter_max - itr > 15)):
			model.W, model.H = shift_factors(model.W, model.H, shifts)

		# compute multiplier for W
		mult_W = _compute_mult_W(data, model)

		# update W
		model.W = np.multiply(model.W, mult_W)
		model.loss_hist += [compute_loss(data, model.W, model.H, shifts)]

		# compute multiplier for H
		mult_H = _compute_mult_H(data, model)

		# update H
		model.H.assign(np.multiply(model.H.shift(0), mult_H))

		model.loss_hist += [compute_loss(data, model.W, model.H, shifts)]
		#model.comb_norm += [compute_comb_reg(model.W, model.num_vars)]
		model.seq_norm += [compute_scfo_reg(data, model.W, model.H, shifts, model._kernel)]

		# renormalize H
		model.W, model.H = renormalize(model.W, model.H)

		# check convergence
		prev_loss, new_loss = model.loss_hist[-2:]
		if (np.abs(prev_loss - new_loss) < model.tol):
			converged = True
			break


def _compute_mult_W(data, model):
	# preallocate
	mult_W = np.zeros(model.W.shape)

	est = model.predict()
	reg_gW = model.l2_scfo * compute_scfo_gW(data, model.W, model.H, 
			   model._shifts, model._kernel)
	#comb_gW = model.l_comb*compute_comb_gW(model.W, model.num_vars)
	for l, t in enumerate(model._shifts):
		num = np.dot(data.shift(0), model.H.shift(t).T)
		denom = np.dot(est, model.H.shift(t).T) + reg_gW[l] + model.l1_W 
		#+ comb_gW[l]
		mult_W[l] = np.divide(num, denom + EPSILON)

	return mult_W


def _compute_mult_H(data, model):
	est = ShiftMatrix(model.predict(), model.maxlag)
	reg_gH = model.l2_scfo * compute_scfo_gH(data, model.W, model.H,
	                                         model._shifts, model._kernel)

	num = tensor_transconv(model.W, data, model._shifts)
	denom = tensor_transconv(model.W, est, model._shifts) + reg_gH + model.l1_H

	return np.divide(num, denom + EPSILON)
