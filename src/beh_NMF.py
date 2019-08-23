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
import pywt

sys.path.append("cnmfpy")
from load_save_data import *
from cwt_utils import get_cwt_scales
from cnmf import CNMF
from conv import ShiftMatrix, vector_conv, shiftVector
from regularize import compute_scfo_reg
from misc import computeNumFactors


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
		self.nFreqs = int(self.metadata['Wavelet']['num_freqs'])
		
		self.subsample = int(self.metadata['Raw data']['subsample'])
		self.nD = int(self.metadata['Raw data']['length'])
		if self.nD == -1:
			self.nD = self.raw_data.shape[0]
		
		self.Tt = self.raw_data[:self.nD*self.subsample:self.subsample, 0]
		self.dt = self.Tt[1] - self.Tt[0]
		self.nT = len(self.Tt)
		
		self.Xx = self.raw_data[:self.nD*self.subsample:self.subsample, 1:]
		self.nVars = self.Xx.shape[1]
		
		self.cwt_matrix = np.empty((self.nFreqs, self.nT, self.nVars))*np.nan
		
	def transform(self):
		"""
		Wavelet transform each variable separately.
		"""
	
		for iV in range(self.nVars):
			scales = get_cwt_scales(self.wavelet, 0.01, .1, self.dt, self.nFreqs)
			coefs, freqs = pywt.cwt(self.Xx[:, iV], scales, self.wavelet, self.dt)			
			self.cwt_matrix[:, :, iV] = abs(coefs)**2.0

		save_cwt_matrix(self.cwt_matrix, self.exp_dir, self.exp_name)
	
	def plot(self):
		"""
		Just quick plot to visualize
		"""
		
		cwt_matrix = load_cwt_matrix(self.exp_dir, self.exp_name)
		for iV in range(self.nVars):
			fig = plt.figure(figsize=(15, 4))
			fig.title('Variable %s' % iV)
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
		self.nVars = self.cwt_matrix.shape[2]
		self.num_patterns = int(self.metadata['NMF']['max_num_patterns'])
		self.pattern_length = int(self.metadata['NMF']['pattern_length'])
		
		min = float(self.metadata['NMF']['seqnmf_norm_min'])
		max = float(self.metadata['NMF']['seqnmf_norm_max'])
		num = int(self.metadata['NMF']['seqnmf_norm_steps'])
		seqnmf_norms = np.geomspace(min, max, num)
		assert seqnmf_norm_idx < num, "seqnmf_norm_idx must be smaller "\
			"than seqnmf_norm_steps in metadata"
		self.seqnmf_norm_idx = seqnmf_norm_idx
		self.seqnmf_norm = seqnmf_norms[seqnmf_norm_idx]

		self.tol = tol
		self.iterations = iterations
		
	def factorize(self):
		"""
		Run a single factorization for a given seqnmf norm, for all variables.
		"""
		
		# To hold NMF results for each variable
		self.NMF_model_list = []
		for iV in range(self.nVars):
			cwt_matrix = self.cwt_matrix[:, :, iV]
			
			model = CNMF(self.num_patterns, self.pattern_length, 
						 l2_scfo=self.seqnmf_norm, l1_W=0, l1_H=0, 
						 n_iter_max=self.iterations, tol=self.tol)
			model.fit(cwt_matrix, alg='mult')
			self.NMF_model_list.append(model)			
		save_NMF_factors(self.NMF_model_list, self.exp_dir, self.exp_name, 
						 self.seqnmf_norm_idx)

	def plot(self):
		"""
		Just plot W, H and reconstructed X for each pattern and variable.
		"""
		
		self.NMF_model_list = load_NMF_factors_single_norm(self.exp_dir, 
								self.exp_name, self.seqnmf_norm_idx)
		for iV in range(self.nVars):
			model = self.NMF_model_list[iV]
			num_factors = computeNumFactors(model)
			
			fig = plt.figure(figsize=(15, 4*num_factors))
			fig.suptitle('Variable %s' % iV)
			gs = GridSpec(num_factors*4, 5)
			for iF in range(num_factors):
				reconstruct_x = vector_conv(model.W[:, :, iF],
				  shiftVector(model.H.shift(0)[iF], model.H.L), model._shifts)
				
				fig.add_subplot(gs[4*iF + 1:4*iF + 3, 0])
				plt.imshow(model.W.T[iF])
				fig.add_subplot(gs[4*iF, 1:-1])
				plt.plot(model.H.shift(0)[iF])
				fig.add_subplot(gs[4*iF + 1: 4*iF + 3, 1:-1])
				plt.imshow(reconstruct_x)
			
			plt.show()
					
			
			
"""
class PCA(object):
	
class cluster(object):
	
"""
