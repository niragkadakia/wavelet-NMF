# -*- coding: utf-8 -*-
"""
Functions for cleaning data that has been loaded from a .pkl file, for fly 
walking analysis.

Helen Cai
10/23/2019
"""
# Libraries: 
# Use base python for now

def get_behaving_trjn(dict):
	"""
	Figure out which trajectories are "behaving"
	
	Args
	-------
	dict: dictionary
		assay data, previously loaded from .pkl
		
	Returns
	-------
	behaving_trjn: dictionary
		true booleans for all trajectory numbers that are behaving
	"""
	
	behaving_trjn = {}
	trjn = dict['trjn']
	behaving = dict['behaving']
	
	for i in range(len(trjn)):
		if behaving_trjn.get(int(trjn[i])) == None:
			if behaving[i]:
				behaving_trjn[int(trjn[i])] = 1
			else:
				pass
		else:
			pass
		
	return behaving_trjn

def get_trj_dict(N, variables, dict):
	"""
	Write a dictionary with variables for a given trajectory number.
	
	Args
	-------
	N: int
		trajectory number of interest	
	variables: list
		variable names (as strings) of interest		
	dict: dictionary
		assay data, previously loaded from .pkl
		
	Returns
	-------
	trj_dict: dictionary
		dictionary with variable names as keys
	"""
	
	trj_dict = {}
	
	for v in variables:
		trj_dict[v] = []
	
	trj_dict['trjn'] = N
		
	for i in range(len(dict['trjn'])):				
		if dict['trjn'][i] == float(N):
			if 'fps' in variables:
				trj_dict['fps']  = dict['fps'][i]
				variables.remove('fps')				
			for var in variables:
				trj_dict[var].append(dict[var][i])
		else: 
			pass
	
	return trj_dict