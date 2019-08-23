import numpy as np
import numpy.linalg as la
from cnmf import CNMF
import matplotlib.pyplot as plt
from scipy.sparse import rand
from conv import shiftVector, vector_conv


def generateGTData(num_factors, num_neurons, maxlag, total_time, factor_sparseness, time_sparseness, noise_mult_factor,
                   random_state=None):
	data = np.zeros((num_neurons, total_time))
	partial_x_list = []
	factor_list = []
	timetrace_list = []
	for i in range(num_factors):
		if random_state:
			temp_random_state = random_state + 3 * i + 44
			tt_random_state = random_state + i ** 2 + 8
		else:
			temp_random_state = None
			tt_random_state = None
		temp_factor = rand(num_neurons, 2 * maxlag + 1, density=factor_sparseness,
		                   random_state=temp_random_state).toarray().T
		temp_timetrace = rand(1, total_time, density=time_sparseness, random_state=tt_random_state).toarray().reshape(
			(total_time,))
		partial_x_temp = vector_conv(temp_factor, shiftVector(temp_timetrace, maxlag), np.arange(maxlag * 2 + 1) - maxlag)
		partial_x_list.append(partial_x_temp)
		data += partial_x_temp
		factor_list.append(temp_factor)
		timetrace_list.append(temp_timetrace)
	data += noise_mult_factor * np.random.rand(num_neurons, total_time)
	return factor_list, timetrace_list, partial_x_list, data


def generateFakeGTData(num_factors, num_neurons, num_fake_neurons, maxlag, total_time, factor_sparseness,
                       time_sparseness, noise_mult_factor, fake_bg_sparseness, fake_bg_factor, fake_fg_sparseness,
                       fake_fg_factor, random_state=None):
	factor_list, timetrace_list, data = generateGTData(num_factors, num_neurons, maxlag, total_time, factor_sparseness,
	                                                   time_sparseness, noise_mult_factor, random_state)
	if random_state:
		bg_random = random_state + 6
		fg_random = random_state + 12
	else:
		bg_random = None
		fg_random = None
	fake_data_bg = fake_bg_factor * rand(num_fake_neurons, total_time, density=fake_bg_sparseness,
	                                     random_state=bg_random).toarray()
	fake_data_fg = fake_fg_factor * rand(num_fake_neurons, total_time, density=fake_fg_sparseness,
	                                     random_state=fg_random)
	fake_data = fake_data_bg+fake_data_fg
	data = np.concatenate((data, fake_data))
	return factor_list, timetrace_list, data


def seq_nmf_data(N, T, L, K, sparsity=0.8):
	"""Creates synthetic dataset for conv NMF

	Args
	----
	N : number of neurons
	T : number of timepoints
	L : max sequence length
	K : number of factors / rank

	Returns
	-------
	data : N x T matrix
	"""

	# low-rank data
	W, H = np.random.rand(N, K), np.random.rand(K, T)
	W[W < sparsity] = 0
	H[H < sparsity] = 0
	lrd = np.dot(W, H)

	# add a random shift to each row
	lags = np.random.randint(0, L, size=N)
	data = np.array([np.roll(row, l, axis=-1) for row, l in zip(lrd, lags)])
	# data = lrd

	return data, W, H


def test_seq_nmf(N=100, T=120, L=10, K=5):
	data, realW, realH = seq_nmf_data(N, T, L, K)
	realH /= la.norm(realH, axis=-1, keepdims=True)

	losses = []
	for k in range(1, 2 * K + 1):
		if (k == K):
			W, H, costhist, loadings, power = seq_nmf(data, K=k, L=2 * L, lam=10 ** (-6), maxiter=200, H_init=realH)
		else:
			W, H, costhist, loadings, power = seq_nmf(data, K=k, L=2 * L, lam=10 ** (-6), maxiter=200)

		losses.append(power)

		if (k == K):
			estH = H
			estW = W

	# Use Munkres algorithm to match rows of H and estH
	# matchcost = 1 - np.dot(realH, estH.T)
	# indices = Munkres().compute(matchcost.copy())
	# _, prm_est = zip(*indices)
	# estH = estH[list(prm_est)]

	# print('Hdiff: ', np.linalg.norm(estH - realH) / np.linalg.norm(realH))
	error = data - _reconstruct(estW, estH)
	print('Percent error: ', la.norm(error) ** 2 / la.norm(data) ** 2)

	# Plot real H vs estimated H
	fig, axes = plt.subplots(2, 1)
	axes[0].imshow(realH / la.norm(realH, axis=-1, keepdims=True))
	axes[1].imshow(estH)

	# Plot reconstruction error
	plt.figure()
	plt.imshow(np.abs(error), cmap='gray')
	plt.colorbar()

	# Plot losses
	plt.figure()
	plt.plot(np.arange(len(losses)) + 1, losses)
	plt.xlabel('rank')
	plt.ylabel('cost')
	plt.show()


if (__name__ == '__main__'):
	data, W, H = seq_nmf_data(100, 300, 10, 2)

	losses = []

	K = 2
	for k in range(1, K + 1):
		model = CNMF(k, 10, tol=0, n_iter_max=1000).fit(data, alg='bcd_const')
		plt.plot(model.loss_hist[1:])
		losses.append(model.loss_hist[-1])

	plt.figure()
	plt.plot(range(1, K + 1), losses)

	plt.figure()
	plt.imshow(model.predict())
	plt.title('Predicted')

	plt.figure()
	plt.imshow(data)

	plt.show()
