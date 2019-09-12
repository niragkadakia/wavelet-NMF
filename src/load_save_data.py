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
	Unpickle the continuous wavelet transform data for all norms
	
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
		Projected PCA data. Shape is ((number of patterns)*
		(seqnmf indices being chosen), length of nonzero indices)
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
		Keys: X_proj: Projected PCA data. Shape is ((number of patterns)*
		(seqnmf indices being chosen), length of nonzero indices). 
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