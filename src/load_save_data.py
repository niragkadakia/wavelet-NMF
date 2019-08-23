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

	filename = '%s/%s/%s_NMF_%d.npy' % (get_data_dir(), exp_dir, exp_name, 
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

	filename = '%s/%s/%s_NMF_%d.npy' % (get_data_dir(), exp_dir, exp_name, 
										seqnmf_norm_idx)
	with gzip.open(filename, 'rb') as f:
		NMF_model_list = pickle.load(f)
	
	return NMF_model_list
