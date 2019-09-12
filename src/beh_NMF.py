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
from cluster_utils import lasso_select

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
	
		for iV in range(self.num_vars):
			scales = get_cwt_scales(self.wavelet, 0.01, .1, 
									self.dt, self.num_freqs)
			coefs, freqs = pywt.cwt(self.Xx[:, iV], scales, 
									self.wavelet, self.dt)		
			self.cwt_matrix[:, :, iV] = abs(coefs)**2.0

		save_cwt_matrix(self.cwt_matrix, self.exp_dir, self.exp_name)
	
	def plot(self):
		"""
		Just quick plot to visualize
		"""
		
		cwt_matrix = load_cwt_matrix(self.exp_dir, self.exp_name)
		for iV in range(self.num_vars):
			fig = plt.figure(figsize=(15, 4))
			plt.title('Variable %s' % iV)
			plt.imshow(cwt_matrix[:, :, iV])
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
		
		self.Xs = load_all_NMF_data(exp_dir, exp_name)['Xs']
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
			nonzero_idxs = np.sum(self.Xs[self.min_idx: self.max_idx, iV], 
								  axis=(-1, -2)) != 0
			
			# Combine Xs from all patterns and all indices, for a given 
			# variable. Combine all values in X matrix as features. 
			# Thus, shape is (patterns*seqnmf_idxs, nT*num_freqs)
			X_flat = np.reshape(self.Xs[self.min_idx: self.max_idx, iV], 
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
		nonzero W,H pair. The plots will allow selection of specific regions, 
		and from these regions will plot the H-vectors and W-vectors for all 
		points in the region.
		"""
		
		def update_selected_H(selected_idxs):
			"""
			Plot H vectors of selected points only.
			"""
			
			ax_h.clear()
			for iS in selected_idxs:
				ax_h.plot(self.nonzero_Hs[iV][iS])
		
		for iV in range(self.num_vars):
			tsne_obj = manifold.TSNE(n_components=2, perplexity=perplexity, 
								 n_iter=iterations, min_grad_norm=grad_norm)
			tsne_proj_Xs = tsne_obj.fit_transform(self.proj_Xs[iV])
			
			fig, (ax_tsne, ax_h) = plt.subplots(2)
			tsne_pts = ax_tsne.scatter(tsne_proj_Xs[:, 0], 
									   tsne_proj_Xs[:, 1], s=10)
			selector = lasso_select(ax_tsne, tsne_pts, update_selected_H)
			plt.show()
		