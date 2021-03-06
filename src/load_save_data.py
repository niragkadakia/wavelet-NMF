"""
Functions for loading and saving data from file.

Created by Nirag Kadakia at 16:12 08-21-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import numpy as np
import os
import configparser
import pickle
import gzip
import datetime
import h5py
import time


def get_data_dir():
	"""
	Get the local data directory from settings.ini
	
	Returns
	-------
	
	data_dir: str
		Absolute path of data directory
	"""	
	
	return load_settings()["Folders"]["data_dir"]
	
def get_analysis_dir():
	"""
	Get the local analysis directory from settings.ini
	
	Returns
	-------
	
	analysis_dir: str
		Absolute path of analysis directory
	"""	
	
	return load_settings()["Folders"]["analysis_dir"]

def get_walk_assay_pkl_data_dir():
	"""
	Get the local directory where pkl objects from the Drosophila 
	smoke walking assay are saved.
	
	Returns
	-------
	
	dir: str
		Absolute path of  directory
	"""	

	return load_settings()["Folders"]["walk_assay_pkl_data_dir"]

def load_settings():
	"""
	Load the configuration file.
	
	Returns
	-------
	
	config: configparser.ConfigParser object
		Loaded configuration of the experiment.
	"""	
	
	config_path = '../data/settings.ini'

	config = configparser.ConfigParser()
	config.read(config_path)

	return config
	

def load_metadata(exp_dir):
	"""
	Load metadata 
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	
	Returns
	-------
	
	metadata: dictionary
		Data for all metadata of the analysis
	"""
	
	config_path = '%s/%s/metadata.ini' % (get_data_dir(), exp_dir)
	
	config = configparser.ConfigParser()
	config.read(config_path)

	return config
	
def load_raw_data(exp_dir, exp_name):
	"""
	Load raw behavioral data traces 
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	
	Returns
	-------
	
	raw_data: numpy array of shape (N, m), m > 1
		Data where first index is timepoint, second is value for 
		time, var1, var2, ...
	"""
	
	filename = '%s/%s/%s.txt' % (get_data_dir(), exp_dir, exp_name)
	raw_data = np.loadtxt(filename)
	
	return raw_data

def load_walk_assay_pkl(pkl_file=None, subdir=None, file=None,
						   assay_type=None):
	"""
	Load pickled experimental matrix file from recorded dataset of the 
	Drosophila smoke odor walking assay.	
	
	Args:
		pkl_file: string of full path for pkl file.
		subdir: string of directory within mahmut_demir/analysis
		file: string of encounter dataset file (-.mat)
		assay_type: structure of .mat to load.
		
		If either subdir or file is supplied, pkl_file will be ignored
			and pkl_file will be created from subdir and file. One of 
			these three must be supplied however.
		
	Returns:
		exp_matrix_dict: Ordered dictionary whose keys are the encounter 
			events, e.g. 'trjn' = trajectory number; 'sx' = signal x-position.
			The dictionary values are arrays whose values are the value of 
			that key at all framenumbers (combined over all trajectories 
			and videos)
	"""
	
	time_start = time.time()
	
	print ('Loading walking assay data...')
	if pkl_file is not None:
		pass
	elif (subdir is not None) and (file is not None):
		out_dir = '%s/%s' % (get_walk_assay_pkl_data_dir(), subdir)
		if assay_type is None:
			pkl_file = '%s/%s.pkl' % (out_dir, file)
		else:
			pkl_file = '%s/%s/%s.pkl' % (out_dir, file, assay_type)
	else:
		print ("Must supply pickle filename in pkl_file or the subdirectory" \
				"and filename in 'subdir' and path'")
		quit()
	
	with open(pkl_file, 'rb') as f:
		exp_matrix_dict = pickle.load(f, encoding='latin1')
	
	print('Loaded pkl file in %.3f seconds' % (time.time() - time_start))
	
	return exp_matrix_dict

def save_cwt_matrix(cwt_matrix, exp_dir, exp_name):
	"""
	Save the continuous wavelet transform data as pickle.
	
	Args
	-------
	
	cwt_matrix: numpy array
		Wavelet data to be saved
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	"""

	filename = '%s/%s/%s_cwt.npy' % (get_data_dir(), exp_dir, exp_name)
	np.save(filename, cwt_matrix)
	
def load_cwt_matrix(exp_dir, exp_name):
	"""
	Save the continuous wavelet transform data as pickle.
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	
	Returns
	-------
	
	cwt_matrix: numpy array of shape (N, T, M)
		wavelet matrices where N is number of frequencies, T is 
		number of timepoints, and M is number of variables
	"""

	filename = '%s/%s/%s_cwt.npy' % (get_data_dir(), exp_dir, exp_name)
	cwt_matrix = np.load(filename)
	
	return cwt_matrix
	
def save_NMF_factors(NMF_model_list, exp_dir, exp_name, seqnmf_norm_idx):
	"""
	Save the continuous wavelet transform data as pickle.
	
	Args
	-------
	
	NMF_model_list: list of CNMF objects of length (num_vars)
		List of CNMF objects, each of which contains W and H factors for each
		behavioral variable.
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	seqnmf_norm_idx: int
		Which seqnmf norm index is being used for estimation. The value of the
		norm defined by this index is given by a linspace of seqnmf_norm_min, 
		seqnmf_norm_max, and seqnmf_norm_steps, which are defined in metadata.
	"""

	filename = '%s/%s/%s_NMF_%d.pkl' % (get_data_dir(), exp_dir, exp_name, 
										seqnmf_norm_idx)
	with gzip.open(filename, 'wb') as f:
		pickle.dump(NMF_model_list,  f)

def load_NMF_factors_single_norm(exp_dir, exp_name, seqnmf_norm_idx):
	"""
	Unpickle the continuous wavelet transform data for one seqnmf norm
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	seqnmf_norm_idx: int
		Which seqnmf norm index is being used for estimation. The value of the
		norm defined by this index is given by a linspace of seqnmf_norm_min, 
		seqnmf_norm_max, and seqnmf_norm_steps, which are defined in metadata.
	
	Returns
	-------
	
	NMF_model_list: list of CNMF objects of length (num_vars)
		List of CNMF objects, each of which contains W and H factors for each
		behavioral variable.
	"""

	filename = '%s/%s/%s_NMF_%d.pkl' % (get_data_dir(), exp_dir, exp_name, 
										seqnmf_norm_idx)
	with gzip.open(filename, 'rb') as f:
		NMF_model_list = pickle.load(f)
	
	return NMF_model_list

def save_all_NMF_data(exp_dir, exp_name, Ws, Hs, Xs, errs):
	"""
	Pickle the continuous wavelet transform data for all norms
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	Ws, Hs, Xs: numpy arrays
		NMF data to be saved for all runs
	"""
	
	filename = '%s/%s/%s_NMF_Ws.npz' % (get_data_dir(), exp_dir, exp_name)
	np.savez_compressed(filename, Ws=Ws)
	filename = '%s/%s/%s_NMF_Hs.npz' % (get_data_dir(), exp_dir, exp_name)
	np.savez_compressed(filename, Hs=Hs)
	filename = '%s/%s/%s_NMF_Xs.npz' % (get_data_dir(), exp_dir, exp_name)
	np.savez_compressed(filename, Xs=Xs)
	filename = '%s/%s/%s_NMF_errs.npz' % (get_data_dir(), exp_dir, exp_name)
	np.savez_compressed(filename, errs=errs)
	
def load_all_NMF_data(exp_dir, exp_name, load_Ws=True, load_Hs=True, 
					  load_Xs=True):
	"""
	Load the NMF data -- W, H, W -- for all seqnmf runs.
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	load_Ws, load_Hs, load_Xs: bool
		Load these if true.
		
	Returns
	-------
	
	data: dictionary
		NMF data saved to file. Contains Ws, Hs, Xs; numpy arrays. 
		For each array, first index is seqnmf norm index, 2nd is 
		variable, 3rd is pattern number, and last idxs are values.
	"""

	Ws = None
	Hs = None
	Xs = None
	
	if load_Ws == True:
		filename = '%s/%s/%s_NMF_Ws.npz' % (get_data_dir(), exp_dir, exp_name)
		Ws = np.load(filename)['Ws']
	if load_Hs == True:
		filename = '%s/%s/%s_NMF_Hs.npz' % (get_data_dir(), exp_dir, exp_name)
		Hs = np.load(filename)['Hs']
	if load_Xs == True:
		filename = '%s/%s/%s_NMF_Xs.npz' % (get_data_dir(), exp_dir, exp_name)
		Xs = np.load(filename)['Xs']
	
	data = dict()
	data['Ws'] = Ws
	data['Hs'] = Hs
	data['Xs'] = Xs
	
	return data
	
def load_NMF_errs(exp_dir, exp_name):
	"""
	Load the reconstruction and regularization errors of NMF data 
	for all seqnmf runs.
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	
	Returns
	-------
	errs: numpy array
		NMF error data saved to file. First index is seqnmf norm index, 2nd is
		variable, 3rd is either reconstruction error (0) or 
		regularization error (1)
	"""

	filename = '%s/%s/%s_NMF_errs.npz' % (get_data_dir(), exp_dir, exp_name)
	errs = np.load(filename)['errs']
	
	return errs
	
def save_PCA_data(exp_dir, exp_name, X_proj, X_nonzero_idxs, 
				  explained_variance):
	"""
	Save PCA data: nonzero-X indices, projected X for nonzero indices, 
	explained variance for each PCA component.
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	X_proj: list of length variables, each element is numpy array
		Projected PCA data. Shape is (length of nonzero indices, 
		number of PCA components)
	X_nonzero_idxs: list of length variables, each element is boolean array
		Shape is (seqnmf indices being chosen, number of patterns), where
		value is 1 if X is nonzero.
	PCA_explained_variance: list of length variables, each element is array
		Each element gives PCA explained ratio for all PCA components
	"""
	
	filename = '%s/%s/%s_PCA_data.npz' % (get_data_dir(), exp_dir, exp_name)
	np.savez(filename, X_proj=X_proj, X_nonzero_idxs=X_nonzero_idxs, 
			explained_variance=explained_variance)

def load_PCA_data(exp_dir, exp_name):
	"""
	Load the PCA data.
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	
	Returns:
	data: dictionary 
		Keys: X_proj: list of length variables, each element is numpy array
		Projected PCA data. Shape is (length of nonzero indices, 
		number of PCA components).
		X_nonzero_idxs: list of length variables, each element is bool array.
		Shape is (seqnmf indices being chosen, number of patterns), where
		value is 1 if X is nonzero.
		explained_variance: PCA explained variance. list of length variables, 
		each element is array. Each element gives PCA explained ratio for 
		all PCA components.
	"""
	
	filename = '%s/%s/%s_PCA_data.npz' % (get_data_dir(), exp_dir, exp_name)
	data = np.load(filename, allow_pickle=True)
	
	return data
	
def save_cluster_data(exp_dir, exp_name, cluster_xys, cluster_type='tsne'):
	"""
	Save data result of clustering data.
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	cluster_xys: list of length variables, each element is numpy array
		Projected clustered data. Shape is (length of nonzero indices, 2)
	cluster_type: str
		Type of clustering being performed (t-SNE, umap etc.)
	"""
	
	filename = '%s/%s/%s_cluster_data_%s.npz' % (get_data_dir(), exp_dir, 
				exp_name, cluster_type)
	np.savez(filename, cluster_xys=cluster_xys)

def load_cluster_data(exp_dir, exp_name, cluster_type='tsne'):
	"""
	Load clustering data.
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	cluster_type: str
		Type of clustering being performed (t-SNE, umap etc.)
	
	Returns:
	-------
	
	data: dictionary
		keys: cluster_xys: list of length variables, each element is numpy array
		Projected clustered data. Shape is (length of nonzero indices, 2)
	"""
	
	filename = '%s/%s/%s_cluster_data_%s.npz' % (get_data_dir(), exp_dir, 
				exp_name, cluster_type)
	data = np.load(filename, allow_pickle=True)
	
	return data
	
def save_2_var_corrs(exp_dir, exp_name, var1, var2, corrs, corr_shifts):
	"""
	Save the continuous wavelet transform data for all norms
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	var1, var2: ints
		Which variables have been correlated.
	corrs: (N, M) numpy array
		max value of small-lag correlations between H-vectors of var1 and var2
	"""
	
	filename = '%s/%s/%s_corr_vars_%d_%d.npz' % (get_data_dir(), exp_dir,
	  exp_name, var1, var2)
	np.savez(filename, corrs=corrs, corr_shifts=corr_shifts)
	
def load_2_var_corrs(exp_dir, exp_name, var1=0, var2=1):
	"""
	Unpickle the continuous wavelet transform data for all norms
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	var1, var2: ints
		Which variables have been correlated.
	
	Returns:
	-------
	
	corrs: (N, M) numpy array
		max value of small-lag correlations between H-vectors of var1 and var2
	"""
	
	filename = '%s/%s/%s_corr_vars_%d_%d.npz' % (get_data_dir(), exp_dir, 
	  exp_name, var1, var2)
	data = np.load(filename, allow_pickle=True)
	
	return data
	
def save_2_var_XWHs(exp_dir, exp_name, var1, var2, cutpoint_idx, 
							  Xs, Ws, Hs, saved_idxs):
	"""
	Save the continuous wavelet transform data for all norms
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	var1, var2: ints
		Which variables have been correlated.
	cutpoint_idx: int
		index of which cutpoint range based on size of correlation
	Xs, Ws, Hs: list s
		arrays to save
	"""
	
	filename = '%s/%s/%s_corr_vars_%d_%d_XWH_iC_%d.npz' % (get_data_dir(), 
	  exp_dir, exp_name, var1, var2, cutpoint_idx)
	np.savez_compressed(filename, Xs=Xs, Ws=Ws, Hs=Hs, saved_idxs=saved_idxs)
	
def load_2_var_XWHs(exp_dir, exp_name, var1, var2, cutpoint_idx):
	"""
	Save the continuous wavelet transform data for all norms
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	var1, var2: ints
		Which variables have been correlated.
	cutpoint_idx: int
		index of which cutpoint range based on size of correlation
	
	Returns:
	-------
	
	Xs: list 
		X arrays loaded for these two variables
	"""
	
	filename = '%s/%s/%s_corr_vars_%d_%d_XWH_iC_%d.npz' % (get_data_dir(), 
	  exp_dir, exp_name, var1, var2, cutpoint_idx)
	data = np.load(filename, allow_pickle=True)
	
	return data
	
def save_2_var_PCA(exp_dir, exp_name, var1, var2, cutpoint_idx, 
							  X_proj):
	"""
	Save PCA-reduced stacked X-vectors for the correlated variables
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	var1, var2: ints
		Which variables have been correlated.
	cutpoint_idx: int
		index of which cutpoint range based on size of correlation
	X_proj: (N, M) array
		Numbeor of X vectors by PCA dimensions.
	"""
	
	filename = '%s/%s/%s_corr_vars_%d_%d_PCA_iC_%d.npz' % (get_data_dir(), 
	  exp_dir, exp_name, var1, var2, cutpoint_idx)
	data = np.savez(filename, X_proj=X_proj)
	
def load_2_var_PCA(exp_dir, exp_name, var1, var2, cutpoint_idx):
	"""
	Save PCA-reduced stacked X-vectors for the correlated variables
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	var1, var2: ints
		Which variables have been correlated.
	cutpoint_idx: int
		index of which cutpoint range based on size of correlation
	
	Returns:
	-------
	
	data: dictionary of data
		Has PCA-projectedd Xs: shape = X vectors by PCA dimensions.
	"""
	
	filename = '%s/%s/%s_corr_vars_%d_%d_PCA_iC_%d.npz' % (get_data_dir(),
	  exp_dir, exp_name, var1, var2, cutpoint_idx)
	data = np.load(filename, allow_pickle=True)
	
	return data
	
def save_2_var_tSNE(exp_dir, exp_name, var1, var2, cutpoint_idx, 
							  X_proj_tSNE):
	"""
	Save tSNE-d X-vectors for the correlated variables
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	var1, var2: ints
		Which variables have been correlated.
	cutpoint_idx: int
		index of which cutpoint range based on size of correlation
	X_proj: (N, M) array
		Numbeor of X vectors by PCA dimensions.
	"""
	
	filename = '%s/%s/%s_corr_vars_%d_%d_tSNE_iC_%d.npz' % (get_data_dir(), 
	  exp_dir, exp_name, var1, var2, cutpoint_idx)
	np.savez(filename, cluster_xys=X_proj_tSNE)
	
def load_2_var_tSNE(exp_dir, exp_name, var1, var2, cutpoint_idx):
	"""
	Save tSNE-d X-vectors for the correlated variables
	
	Args
	-------
	
	exp_dir: str
		Name of experiment subdirectory within data_dir
	exp_name: str
		Name of .txt file within exp_dir containing data. Should be 
		tab-delimited data whose columns are (time, var1, var2,...)
		and whose rows are the values at each time.
	var1, var2: ints
		Which variables have been correlated.
	cutpoint_idx: int
		index of which cutpoint range based on size of correlation
	
	Returns:
	-------
	
	data: dictionary holding tSNE x-ys
	"""
	
	filename = '%s/%s/%s_corr_vars_%d_%d_tSNE_iC_%d.npz' % (get_data_dir(), 
	  exp_dir, exp_name, var1, var2, cutpoint_idx)
	data = np.load(filename, allow_pickle=True)
		
	return data
	