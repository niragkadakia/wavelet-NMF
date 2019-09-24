"""

Classes for all beh_NMF functions.

Created by Nirag Kadakia at 15:57 08-21-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn import decomposition
from sklearn import manifold
import pywt

from load_save_data import *
from cwt_utils import get_cwt_scales
from NMF_utils import compute_num_factors
from cluster_utils import lasso_select, hover_select

sys.path.append("cnmfpy")
from cnmf import CNMF
from conv import ShiftMatrix, vector_conv, shiftVector
from regularize import compute_scfo_reg


class wavelet_transform(object):
	"""
	Class for calculating wavelet transform of the data
	"""
	
	def __init__(self, exp_dir='ML_test1', exp_name='0'):
		"""
		Initialize data experiment folder, metadata, load behavioral data
		"""
		
		self.exp_dir = exp_dir
		self.exp_name = exp_name
		self.metadata = load_metadata(exp_dir)
		self.raw_data = load_raw_data(exp_dir, exp_name)
		assert len(self.raw_data.shape) == 2,"Data must be N rows "\
		  "by 2+ columns; first column is time."
		
		self.wavelet = self.metadata['Wavelet']['name']
		self.num_freqs = int(self.metadata['Wavelet']['num_freqs'])
		self.max_freq = float(self.metadata['Wavelet']['max_freq'])
		self.min_freq = float(self.metadata['Wavelet']['min_freq'])
		
		self.subsample = int(self.metadata['Raw data']['subsample'])
		length_data = int(self.metadata['Raw data']['length'])
		if length_data == -1:
			length_data = self.raw_data.shape[0]
		
		self.Tt = self.raw_data[:length_data*self.subsample:self.subsample, 0]
		self.dt = self.Tt[1] - self.Tt[0]
		self.nT = len(self.Tt)
		
		self.Xx = self.raw_data[:length_data*self.subsample:self.subsample, 1:]
		self.num_vars = self.Xx.shape[1]
		
		self.cwt_matrix = np.empty((self.num_freqs, self.nT, 
									self.num_vars))*np.nan
		
	def transform(self):
		"""
		Wavelet transform each variable separately.
		"""
	
		scales = get_cwt_scales(self.wavelet, self.min_freq, self.max_freq,
									self.dt, self.num_freqs)
		for iV in range(self.num_vars):
			coefs, freqs = pywt.cwt(self.Xx[:, iV], scales, 
									self.wavelet, self.dt)		
			self.cwt_matrix[:, :, iV] = abs(coefs)**2.0

		save_cwt_matrix(self.cwt_matrix, self.exp_dir, self.exp_name)
	
	def plot(self):
		"""
		Just quick plot to visualize
		"""
		
		scales = get_cwt_scales(self.wavelet, self.min_freq, self.max_freq,
									self.dt, self.num_freqs)
		cwt_matrix = load_cwt_matrix(self.exp_dir, self.exp_name)
		for iV in range(self.num_vars):
			fig, ax = plt.subplots(2, 1)
			fig.set_size_inches(15, 4)
			plt.suptitle('Variable %s' % iV)
			
			ax[0].plot(self.Tt, self.Xx[:, iV])
			ax[0].set_xlim(self.Tt[0], self.Tt[-1])
			
			y, x = np.meshgrid(np.log(scales[::-1])/np.log(10), self.Tt)
			ax[1].set_xlim(self.Tt[0], self.Tt[-1])
			ax[1].pcolormesh(x, y, cwt_matrix[:, :, iV].T)
			plt.ylabel('log freq') 
			plt.xlabel('Time')
			plt.show()
		

class random_proj_transform(object):
	"""
	Class for calculating wavelet transform of the data
	"""
	
	def __init__(self, exp_dir='ML_test4', exp_name='0'):
		"""
		Initialize data experiment folder, metadata, load behavioral data
		"""
		
		## TODO This class is still under construction
		
		self.exp_dir = exp_dir
		self.exp_name = exp_name
		self.metadata = load_metadata(exp_dir)
		self.raw_data = load_raw_data(exp_dir, exp_name)
		assert len(self.raw_data.shape) == 2,"Data must be N rows "\
		  "by 2+ columns; first column is time."
		
		self.num_projs = int(self.metadata['Random Projection']['num_projs'])
		
		self.subsample = int(self.metadata['Raw data']['subsample'])
		length_data = int(self.metadata['Raw data']['length'])
		if length_data == -1:
			length_data = self.raw_data.shape[0]
		
		self.Tt = self.raw_data[:length_data*self.subsample:self.subsample, 0]
		self.dt = self.Tt[1] - self.Tt[0]
		self.nT = len(self.Tt)
		
		self.Xx = self.raw_data[:length_data*self.subsample:self.subsample, 1:]
		
		# Data variabls is num_raw_vars. However, only transformed data only 
		# uses one variable that is a lin.comb. of the original space.
		self.num_raw_vars = self.Xx.shape[1]
		self.num_vars = 1
		
		self.cwt_matrix = np.empty((self.num_projs, self.nT, 
		  self.num_vars))*np.nan
		
	def transform(self):
		"""
		Random rotation (SO(n)) of variables into new space.
		"""
	
		from scipy.stats import special_ortho_group
		for iF in range(self.num_projs):
			R = special_ortho_group.rvs(self.num_raw_vars)
			rot_xy = (np.dot(R, self.Xx.T))**2.0
			self.cwt_matrix[iF, :, 0] = rot_xy.T[:, 0]
		save_cwt_matrix(self.cwt_matrix, self.exp_dir, self.exp_name)
	
	def plot(self):
		"""
		Just quick plot to visualize
		"""
		
		cwt_matrix = load_cwt_matrix(self.exp_dir, self.exp_name)
		for iV in range(self.num_vars):
			fig, ax = plt.subplots(2, 1)
			fig.set_size_inches(15, 4)
			plt.suptitle('Variable %s' % iV)
			ax[0].plot(self.Tt, self.Xx[:, iV])
			ax[1].imshow(cwt_matrix[:, :, iV])
			plt.show()
		
		
class NMF(object):
	"""
	Factorize the wavelet-tranformed data using CNMF/SEQNMF
	"""
	
	def __init__(self, exp_dir='ML_test1', exp_name='0', seqnmf_norm_idx=0, 
				 iterations=2000, tol=1e-7):
		"""
		Initialize data experiment folder, metadata, load wavelet data
		"""
		
		self.exp_dir = exp_dir
		self.exp_name = exp_name
		self.metadata = load_metadata(exp_dir)
		self.cwt_matrix = load_cwt_matrix(exp_dir, exp_name)
		self.num_vars = self.cwt_matrix.shape[2]
		self.num_max_patterns = int(self.metadata['NMF']['max_num_patterns'])
		self.pattern_length = int(self.metadata['NMF']['pattern_length'])
		
		min = float(self.metadata['NMF']['seqnmf_norm_min'])
		max = float(self.metadata['NMF']['seqnmf_norm_max'])
		num = int(self.metadata['NMF']['seqnmf_norm_steps'])
		self.seqnmf_norms = np.geomspace(min, max, num)
		assert seqnmf_norm_idx < num, "seqnmf_norm_idx must be smaller "\
			"than seqnmf_norm_steps in metadata"
		self.seqnmf_norm_idx = seqnmf_norm_idx
		self.seqnmf_norm = self.seqnmf_norms[seqnmf_norm_idx]

		self.tol = tol
		self.iterations = iterations
		
	def factorize(self):
		"""
		Run a single factorization for a given seqnmf norm, for all variables.
		"""
		
		# To hold NMF results for each variable
		self.NMF_model_list = []
		for iV in range(self.num_vars):
			cwt_matrix = self.cwt_matrix[:, :, iV]
			
			model = CNMF(self.num_max_patterns, self.pattern_length, 
						 l2_scfo=self.seqnmf_norm, l1_W=0, l1_H=0, 
						 n_iter_max=self.iterations, tol=self.tol)
			model.fit(cwt_matrix, alg='mult')
			self.NMF_model_list.append(model)			
		save_NMF_factors(self.NMF_model_list, self.exp_dir, self.exp_name, 
						 self.seqnmf_norm_idx)

	def plot(self):
		"""
		Just plot W, H and reconstructed X for each pattern and variable, for
		one seqnmf norm.
		"""
		
		self.NMF_model_list = load_NMF_factors_single_norm(self.exp_dir, 
								self.exp_name, self.seqnmf_norm_idx)
		for iV in range(self.num_vars):
			model = self.NMF_model_list[iV]
			num_nonzero_patterns = compute_num_factors(model)
			
			fig = plt.figure(figsize=(15, 4*num_nonzero_patterns))
			fig.suptitle('Variable %s' % iV)
			gs = GridSpec(num_nonzero_patterns*4, 5)
			for iF in range(num_nonzero_patterns):
				reconstruct_x = vector_conv(model.W[:, :, iF],
				  shiftVector(model.H.shift(0)[iF], model.H.L), model._shifts)
				
				fig.add_subplot(gs[4*iF + 1:4*iF + 3, 0])
				plt.imshow(model.W.T[iF])
				fig.add_subplot(gs[4*iF, 1:-1])
				plt.plot(model.H.shift(0)[iF])
				fig.add_subplot(gs[4*iF + 1: 4*iF + 3, 1:-1])
				plt.imshow(reconstruct_x)
			
			plt.show()

	def agg_data(self):
		"""
		Aggregate the NMF data into a single file for ease of loading.
		Also calculate reconstruction and regularization errors for 
		each index and variable.
		"""
		
		cwt_matrix = load_cwt_matrix(self.exp_dir, self.exp_name)
		NMF_idxs = range(int(self.metadata['NMF']['seqnmf_norm_steps']))
		Ws = None
		Hs = None
		Xs = None
		errs = np.empty((len(NMF_idxs), self.num_vars, 2))*np.nan
		
		for iR in NMF_idxs:
			print (iR)
			NMF_model_list = load_NMF_factors_single_norm(self.exp_dir, 
								self.exp_name, iR)
			if Ws is None:
				W_shape = (len(NMF_idxs), self.num_vars, 
						   self.num_max_patterns) + \
						   NMF_model_list[0].W[:, :, 0].shape
				Ws = np.empty(W_shape)*np.nan
			if Hs is None:
				H_shape = (len(NMF_idxs), self.num_vars, 
						   self.num_max_patterns) + \
						   NMF_model_list[0].H.shift(0)[0].shape
				Hs = np.empty(H_shape)*np.nan
			if Xs is None:
				X_shape = (len(NMF_idxs), self.num_vars, 
						   self.num_max_patterns, 
						   W_shape[-1], H_shape[-1])
				Xs = np.empty(X_shape)*np.nan
				
			for iV in range(self.num_vars):
			
				# Get W, H, X for each pattern. Full X is sum over patterns.
				for iP in range(self.num_max_patterns):
					model = NMF_model_list[iV]
					Ws[iR, iV, iP] = model.W[:, :, iP]
					Hs[iR, iV, iP] = model.H.shift(0)[iP]
					Xs[iR, iV, iP] = vector_conv(Ws[iR, iV, iP], 
					  shiftVector(model.H.shift(0)[iP], model.H.L), 
					  model._shifts)
				
				norm = np.linalg.norm(cwt_matrix[:, :, iV])
				reconstruct_err = np.linalg.norm(cwt_matrix[:, :, iV] - 
									np.sum(Xs[iR, iV], axis=0))/norm
				regularize_err = compute_scfo_reg(ShiftMatrix(
									cwt_matrix[:, :, iV], 
									self.pattern_length), model.W, model.H,	
									model._shifts, model._kernel)/norm**2
				errs[iR, iV, 0] = reconstruct_err
				errs[iR, iV, 1] = regularize_err
			
		save_all_NMF_data(self.exp_dir, self.exp_name, Ws, Hs, Xs, errs)
		
	def plot_errs(self):
		"""
		Plot the reconstruction and regularization errors
		"""
		
		errs = load_NMF_errs(self.exp_dir, self.exp_name)
		for iV in range(self.num_vars):
			rec_errs = errs[:, iV, 0]/max(errs[:, iV, 0])
			reg_errs = errs[:, iV, 1]/max(errs[:, iV, 1])
			fig = plt.figure(figsize=(8, 3))
			plt.title('Errors')
			plt.plot(self.seqnmf_norms,  rec_errs, label='rec')
			plt.plot(self.seqnmf_norms,  reg_errs, label='reg')
			plt.xscale('log')
			plt.show()
			
			
class PCA(object):
	"""
	PCA transform the NMF returned patterns, for a given range of regularizer.
	PCA is done on the reconstructed data matrices.
	"""
	
	def __init__(self, exp_dir='ML_test1', exp_name='0'):
		"""
		Initialize data experiment folder, metadata, load NMF data, set
		bounds for which seqnmf values to use. 
		"""
		
		self.exp_dir = exp_dir
		self.exp_name = exp_name
		self.metadata = load_metadata(exp_dir)
		
		# Indices for which to aggregate data to do PCA reduction.
		self.min_idx = int(self.metadata['PCA']['seqnmf_norm_min_idx'])
		self.max_idx = int(self.metadata['PCA']['seqnmf_norm_max_idx'])
		self.num_components = int(self.metadata['PCA']['num_components'])
		self.num_idxs = self.max_idx - self.min_idx
		
		data_dict = load_all_NMF_data(exp_dir, exp_name, 
		  load_Xs=True, load_Ws=True)
		self.Xs = data_dict['Xs']
		self.Ws = data_dict['Ws']
		self.pattern_length = self.Ws.shape[-2]
		self.num_vars = self.Xs.shape[1]
		self.num_max_patterns = self.Xs.shape[2]
		
	def transform_PCA(self):
		"""
		Do the PCA dimensionality reduction
		"""
		
		self.X_proj = []
		self.X_nonzero_idxs = []
		self.explained_variance = []
		for iV in range(self.num_vars):
			
			# Only grab indices for which at least one X is nonzero
			# Discount patterns that only appear at boundaries.
			nonzero_idxs = np.sum(self.Xs[self.min_idx: self.max_idx, iV, :, :,
								  self.pattern_length:-self.pattern_length],
								  axis=(-1, -2)) > 1e-5
			
			# Combine Xs from all patterns and all indices, for a given 
			# variable. Combine all values in X matrix as features. 
			# Thus, shape is (patterns*seqnmf_idxs, nT*num_freqs)
			# Also, remove boundaries containing edges -- these have
			# boundary effects that should be ignored.
			X_flat = np.reshape(self.Xs[self.min_idx: self.max_idx, iV, :, :,
								self.pattern_length:-self.pattern_length],
								(self.num_max_patterns*self.num_idxs, -1))
			X_flat_nonzero = X_flat[nonzero_idxs.flatten()]
			
			_PCA = decomposition.PCA(n_components=self.num_components)
			_PCA.fit(X_flat_nonzero)
			_X_proj = _PCA.transform(X_flat_nonzero)
			
			self.X_proj.append(_X_proj)
			self.X_nonzero_idxs.append(nonzero_idxs)
			self.explained_variance.append(_PCA.explained_variance_ratio_)
			print (_PCA.explained_variance_ratio_.sum())
			
		save_PCA_data(self.exp_dir, self.exp_name, self.X_proj, 
					  self.X_nonzero_idxs, self.explained_variance)


class cluster(object):
	"""
	t-SNE clustering for all NMF patterns in the given range of regularizers.
	"""
	
	def __init__(self, exp_dir='ML_test1', exp_name='0'):
		"""
		Initialize data experiment folder, metadata, load PCA data, set
		bounds for which seqnmf values to use. 
		"""
		
		self.exp_dir = exp_dir
		self.exp_name = exp_name
		self.metadata = load_metadata(exp_dir)
		
		# Indices for which to aggregate data to do PCA reduction.
		self.PCA_data = load_PCA_data(self.exp_dir, self.exp_name)
		self.proj_Xs = self.PCA_data['X_proj']
		self.nonzero_idxs = self.PCA_data['X_nonzero_idxs']
		self.num_vars = len(self.proj_Xs)
		
		# Load Ws and Hs to compare
		NMF_data = load_all_NMF_data(exp_dir, exp_name, load_Xs=False)
		
		# Get nonzero values of Hs and Ws in lambda-reduced range
		min_idx = int(self.metadata['PCA']['seqnmf_norm_min_idx'])
		max_idx = int(self.metadata['PCA']['seqnmf_norm_max_idx'])		
		self.Ws = NMF_data['Ws'][min_idx: max_idx]
		self.Hs = NMF_data['Hs'][min_idx: max_idx]
		self.nonzero_Hs = []
		self.nonzero_Ws = []
		for iV in range(self.num_vars):
			_H = self.Hs[:, iV, :, :]
			_W = self.Ws[:, iV, :, :, :]
			self.nonzero_Hs.append(_H[self.nonzero_idxs[iV]])
			self.nonzero_Ws.append(_W[self.nonzero_idxs[iV]])
		
		# Number of nonzero patterns (points to be plotted)
		self.num_pts = self.proj_Xs[0].shape[0]
		self.nT = _H.shape[-1]
		
	def transform_tSNE(self, perplexity=30, iterations=2000, grad_norm=1e-7):
		"""
		t-SNE reduce the PCA-reduced, flattened X-components of each 
		nonzero W,H pair.
		"""
		
		tSNE_data = []
		for iV in range(self.num_vars):
			tsne_obj = manifold.TSNE(n_components=2, perplexity=perplexity, 
								 n_iter=iterations, min_grad_norm=grad_norm)
			tsne_proj_Xs = tsne_obj.fit_transform(self.proj_Xs[iV])
			tSNE_data.append(tsne_proj_Xs)
			
		save_cluster_data(self.exp_dir, self.exp_name, tSNE_data, 
						  cluster_type='tsne')
			
	def plot_Hs_in_region(self, cluster_type='tsne'):
		"""
		The plots will allow selection of specific regions, 
		and from these regions will plot the H-vectors for all points in the 
		region.
		"""
		
		data = load_cluster_data(self.exp_dir, self.exp_name, 
								 cluster_type=cluster_type)
		cluster_xys = data['cluster_xys']
		
		def update_selected_Hs(selected_idxs):
			"""
			Plot H vectors of selected points only.
			"""
			
			ax_h.clear()
			for iS in selected_idxs:
				ax_h.plot(self.nonzero_Hs[iV][iS])
		
		for iV in range(self.num_vars):
			fig, (ax_tsne, ax_h) = plt.subplots(2)
			tsne_pts = ax_tsne.scatter(cluster_xys[iV][:, 0], 
									   cluster_xys[iV][:, 1], s=10)
			selector = lasso_select(ax_tsne, tsne_pts, update_selected_Hs)
			plt.show()
		
	def plot_single_WH(self, cluster_type='tsne'):
		"""
		This plot will allow selection of individual points and show the 
		W-pattern and H-corresponding to it.
		"""
		
		data = load_cluster_data(self.exp_dir, self.exp_name, 
								 cluster_type=cluster_type)
		cluster_xys = data['cluster_xys']
		
		def update_selected_WH(idx):
			"""
			Plot W and H of single points only, hovered over with mouse.
			"""
			
			ax_h.clear()
			ax_h.plot(self.nonzero_Hs[iV][idx].T)
			ax_w.clear()
			ax_w.imshow(self.nonzero_Ws[iV][idx].T)
			
		for iV in range(self.num_vars):
			fig, (ax_tsne, ax_h, ax_w) = plt.subplots(3)
			tsne_pts = ax_tsne.scatter(cluster_xys[iV][:, 0], 
									   cluster_xys[iV][:, 1], s=10)
			selector = hover_select(fig, ax_tsne, tsne_pts, update_selected_WH)
			fig.canvas.mpl_connect("motion_notify_event", selector.hover)
			plt.show()


class correlate_multiple_vars(object):
	"""
	Use correlation analysis to analyze the data for multiple variables,
	each of which is wavelet transformed.
	
	Uses parameters of PCA to get the minimum and maximum norms to use for
	correlation.
	"""
	
	def __init__(self, exp_dir='ML_test2', exp_name='0', corr_vars=[0, 1],
				 num_max_corr_patterns=5000):
		"""
		Initialize data experiment folder, metadata, load PCA data, set
		bounds for which seqnmf values to use. 
		"""
		
		self.exp_dir = exp_dir
		self.exp_name = exp_name
		self.metadata = load_metadata(exp_dir)
		
		# Indices for nonzero Xs -- use PCA to get nonzero vals and ranges
		self.PCA_data = load_PCA_data(self.exp_dir, self.exp_name)
		self.nonzero_idxs = self.PCA_data['X_nonzero_idxs']
		
		# Load Ws and Hs to compare
		NMF_data = load_all_NMF_data(exp_dir, exp_name, load_Xs=False)
		
		# Indices of variables to correlate
		self.var1 = corr_vars[0]
		self.var2 = corr_vars[1]
		
		# Max number of pairs to use for PCA
		self.num_max_corr_patterns = num_max_corr_patterns
		
		# Get nonzero values of Hs and Ws in lambda-reduced range
		min_idx = int(self.metadata['PCA']['seqnmf_norm_min_idx'])
		max_idx = int(self.metadata['PCA']['seqnmf_norm_max_idx'])		
		self.Ws = NMF_data['Ws'][min_idx: max_idx]
		self.Hs = NMF_data['Hs'][min_idx: max_idx]
		self.num_vars = self.Ws.shape[1]
		
		self.nonzero_Hs = []
		self.nonzero_Ws = []
		for iV in range(self.num_vars):
			_H = self.Hs[:, iV, :, :]
			_W = self.Ws[:, iV, :, :, :]
			self.nonzero_Hs.append(_H[self.nonzero_idxs[iV]])
			self.nonzero_Ws.append(_W[self.nonzero_idxs[iV]])
		
	def calc_2_var_corrs(self):
		"""
		Get the maximum cross-correlation in a small time lag between 
		the H vectors of two data variables, for all possible combinations
		of H_1 and H_2.
		"""
		
		num_var1_patterns = len(self.nonzero_Hs[self.var1])
		num_var2_patterns = len(self.nonzero_Hs[self.var2])
		
		corrs = np.empty((num_var1_patterns, num_var2_patterns))*np.nan
		corr_shifts = np.empty((num_var1_patterns, num_var2_patterns))*np.nan
		
		for iP in range(num_var1_patterns):
			print (iP, 'of', num_var1_patterns)
			for jP in range(num_var2_patterns):
				
				H1 = self.nonzero_Hs[self.var1][iP]
				H2 = self.nonzero_Hs[self.var2][jP]

				W1 = self.nonzero_Ws[self.var1][iP].T
				W2 = self.nonzero_Ws[self.var2][jP].T
				
				# Absolute value of max correlation lag
				corr_len = W1.shape[1]
				H1 /= np.mean(H1)
				H2 /= np.mean(H2)
				
				corr = np.correlate(H2, H1[corr_len:-corr_len], mode='valid')
				corrs[iP, jP] = max(corr)
				corr_shifts[iP, jP] = corr_len - np.argmax(corr)
		corr_shifts = corr_shifts.astype('int')
		
		save_2_var_corrs(self.exp_dir, self.exp_name, 
		  self.var1, self.var2, corrs, corr_shifts)
		
	def plot_2_var_corrs_hist(self, bins=200, min=10):
		"""
		Plot histogram of all max-cross-correlations between two variables.
		"""
		
		_data = load_2_var_corrs(self.exp_dir, self.exp_name, 
								 self.var1, self.var2)
		corrs = (_data['corrs']).flatten()
		fig = plt.figure(figsize=(10, 5))
		plt.hist(corrs[corrs > min], bins=bins)
		plt.show()
		
	def calc_2_vars_XWHs(self):
		"""
		Stack the wavelet transforms for pairs of factorizations between
		two variables, where the H-vectors of the factorizations have a 
		given correlation.
		"""
	
		## Rename this funcs
	
		_data = load_2_var_corrs(self.exp_dir, self.exp_name, 
								 self.var1, self.var2)
		corr_shifts = _data['corr_shifts']
		corrs = _data['corrs']
		num_var1_patterns = corrs.shape[0]
		num_var2_patterns = corrs.shape[1]
		
		# Get correlation ranges from metadata -- this is pretty ugly but quick
		cutpoints_str = self.metadata['2-var Correlations']['cutpoints']
		cutpoints = np.array(list(filter(None, cutpoints_str.split("\n"))), 
		  dtype='int')
		
		# Load all nonzero Xs for the two variables; delete data array for mem
		_data = load_all_NMF_data(self.exp_dir, self.exp_name, load_Xs=False, 
								  load_Ws=True, load_Hs=True)
		_H = _data['Hs'][:, self.var1, :, :]
		nonzero_Hs_1 = _H[self.nonzero_idxs[self.var1]]
		_H = _data['Hs'][:, self.var2, :, :]
		nonzero_Hs_2 = _H[self.nonzero_idxs[self.var2]]
		_W = _data['Ws'][:, self.var1, :, :, :]
		nonzero_Ws_1 = _W[self.nonzero_idxs[self.var1]]
		_W = _data['Ws'][:, self.var2, :, :, :]
		nonzero_Ws_2 = _W[self.nonzero_idxs[self.var2]]
		nFreqs = nonzero_Ws_1.shape[-1]*2
		
		del (_data)
		del (_H)
		del (_W)
		
		var1_idxs = np.random.choice(num_var1_patterns, size=1000000)
		var2_idxs = np.random.choice(num_var2_patterns, size=1000000)
		idxs = np.vstack((var1_idxs, var2_idxs)).T
		
		for iC in range(len(cutpoints) - 1):
			Xs = []
			Hs = []
			Ws = []
			saved_idxs = []
			corr_lo = cutpoints[iC]
			corr_hi = cutpoints[iC + 1]
			for iP, jP in idxs:
				print (len(Xs), '...', iP, jP)
				if (corrs[iP, jP] > corr_hi) or \
				  (corrs[iP, jP] < corr_lo):
					continue
				
				H1 = nonzero_Hs_1[iP]
				H2 = nonzero_Hs_2[jP]
				W1 = nonzero_Ws_1[iP]
				W2 = nonzero_Ws_2[jP]
				
				idx = corr_shifts[iP, jP]
				
				# Shift H2 to align to max correlation with H1; backshift W2
				H2_shift = np.roll(H2, idx)
				W2_shift = np.roll(W2, -idx, axis=0)
				if idx < 0:
					H2_shift[idx:] = 0
					W2_shift[idx:] = 0
				elif idx >= 0:
					H2_shift[:idx] = 0
					W2_shift[:idx] = 0
				H = H1*H2_shift
				W = np.vstack((W1.T, W2_shift.T))
				#plt.imshow(W)
				#plt.show()
				X = np.zeros((nFreqs, len(H)))
				for iF in range(nFreqs):
					X[iF] = np.convolve(W[iF], H, mode='same')
				#plt.imshow(X)
				#plt.show()
				
				Xs.append(X.astype(np.float16))
				Ws.append(W)
				Hs.append(H)
				saved_idxs.append([iP, jP])
				
				if len(Xs) >= self.num_max_corr_patterns:
					break	
								
			saved_idxs = np.asarray(saved_idxs).astype(np.int)
			Xs = np.asarray(Xs)
			Ws = np.asarray(Ws)
			Hs = np.asarray(Hs)
			save_2_var_XWHs(self.exp_dir, self.exp_name, 
									  self.var1, self.var2, iC, Xs, 
									  Ws, Hs, saved_idxs)
	
	def PCA(self):
		"""
		Initialize data experiment folder, metadata, load NMF data, set
		bounds for which seqnmf values to use. 
		"""
		
		# Get correlation ranges from metadata -- this is pretty ugly but quick
		cutpoints_str = self.metadata['2-var Correlations']['cutpoints']
		cutpoints = np.array(list(filter(None, cutpoints_str.split("\n"))), 
		  dtype='int')
		
		# Indices for which to aggregate data to do PCA reduction.
		self.num_components = int(self.metadata['PCA']['num_components'])
		
		for iC in range(len(cutpoints) - 1):
			
			_data = load_2_var_XWHs(self.exp_dir, self.exp_name, 
					  self.var1, self.var2, iC)
			Xs = _data['Xs']
			Xs_flat = np.reshape(Xs, (Xs.shape[0], -1))

			_PCA = decomposition.PCA(n_components=self.num_components)
			_PCA.fit(Xs_flat)
			X_proj = _PCA.transform(Xs_flat)
			print (_PCA.explained_variance_ratio_.sum())
			
			save_2_var_PCA(self.exp_dir, self.exp_name, 
									  self.var1, self.var2, iC, X_proj)
									  
	def transform_tSNE(self, perplexity=100, iterations=2000, grad_norm=1e-7):
		"""
		t-SNE reduce the PCA-reduced, flattened X-components of each 
		nonzero W,H pair.
		"""
		
		# Get correlation ranges from metadata -- this is pretty ugly but quick
		cutpoints_str = self.metadata['2-var Correlations']['cutpoints']
		cutpoints = np.array(list(filter(None, cutpoints_str.split("\n"))), 
		  dtype='int')
		
		for iC in range(len(cutpoints) - 1):
			data = load_2_var_PCA(self.exp_dir, self.exp_name, 
											 self.var1, self.var2, iC)
			proj_Xs = data['X_proj']
			tsne_obj = manifold.TSNE(n_components=2, perplexity=perplexity, 
								 n_iter=iterations, min_grad_norm=grad_norm)
			tsne_proj_Xs = tsne_obj.fit_transform(proj_Xs)
			save_2_var_tSNE(self.exp_dir, self.exp_name, self.var1, 
								 self.var2, iC, tsne_proj_Xs)
			
	def plot_single_WH(self, cluster_type='tsne', cutpoint_idx=0):
		"""
		This plot will allow selection of individual points and show the 
		W-pattern and H-corresponding to it.
		"""
		
		
		self.raw_data = load_raw_data(self.exp_dir, self.exp_name)
		self.subsample = int(self.metadata['Raw data']['subsample'])
		length_data = int(self.metadata['Raw data']['length'])
		if length_data == -1:
			length_data = self.raw_data.shape[0]
		self.Xx = self.raw_data[:length_data*self.subsample:self.subsample, 1:]
		self.X_var1 = self.Xx[:, self.var1]
		self.X_var2 = self.Xx[:, self.var2]
		self.Tt = self.raw_data[:length_data*self.subsample:self.subsample, 0]
		self.dt = self.Tt[1] - self.Tt[0]
		self.nT = len(self.Tt)
	
		_data = load_2_var_tSNE(self.exp_dir, self.exp_name, 0, 1, 0)
		cluster_xys = _data['cluster_xys']
		_data = load_2_var_XWHs(self.exp_dir, self.exp_name, 
					  self.var1, self.var2, cutpoint_idx)
		self.Ws = _data['Ws']
		self.Hs = _data['Hs']
		
		def update_selected_WH(idx):
			"""
			Plot W and H of single points only, hovered over with mouse.
			"""
			
			ax_h.clear()
			ax_h.plot(self.Hs[idx])
			ax_w.clear()
			
			ax_w.set_xlim(0, 500)
			ax_w.pcolormesh(self.Ws[idx][::-1])
			
			ax_phase.clear()
			color = self.Hs[idx]/max(self.Hs[idx])
			idx_to_plot = color > 0
			ax_phase.scatter(self.X_var1[idx_to_plot], self.X_var2[idx_to_plot], 
				c=color[idx_to_plot], s=15, alpha=0.9)
			ax_phase.set_xlim(min(self.X_var1), max(self.X_var1))
			ax_phase.set_ylim(min(self.X_var2), max(self.X_var2))
			
			#ax_raw.set_xlim(self.Tt[0], self.Tt[-1])
			from scipy.signal import find_peaks
			ax_raw.clear()
			ax_raw2.clear()
			peaks, _ = find_peaks(self.Hs[idx])
			wind = 25
			
			## Below is incorrect -- chagen endpoints			
			for peak in peaks:
			
				if self.Hs[idx][peak]/max(self.Hs[idx]) < 0.5:
					continue
				if peak < wind:
					continue
				else:
					min_pk = peak - wind
				if peak > self.nT - wind:
					continue
				else:
					max_pk = peak + wind
				ax_raw.plot(self.X_var1[min_pk: max_pk])
				ax_raw2.plot(self.X_var2[min_pk: max_pk])
				
			
			
		fig, (ax_tsne, ax_h, ax_w, ax_phase, ax_raw, ax_raw2) = plt.subplots(6)
		tsne_pts = ax_tsne.scatter(cluster_xys[:, 0], 
								   cluster_xys[:, 1], s=10)
		selector = hover_select(fig, ax_tsne, tsne_pts, update_selected_WH)
		fig.canvas.mpl_connect("motion_notify_event", selector.hover)
		plt.show()


