# -*- coding: utf-8 -*-
"""
Functions for cleaning data that has been loaded from a .pkl file, for fly 
walking analysis.

Helen Cai
10/23/2019
"""
# Libraries: 
# Use base python for now

def get_behaving_trjn(trjn, behaving):
	"""
	Figure out which trajectories are "behaving"
	
	Args
	-------
	trjn: float64
		trajectory numbers, previously loaded from .pkl
		
	behaving: bool
		indicates if a fly is behaving, previously loaded from .pkl
		
	Returns
	-------

	behaving_trjn: dictionary
		true booleans for all trajectory numbers that are behaving
	"""
	
	behaving_trjn = {}
	
	for i in range(len(trjn)):
		if behaving_trjn.get(int(trjn[i])) == None:
			if behaving[i]:
				behaving_trjn[int(trjn[i])] = 1
			else:
				pass
		else:
			pass
		
	return behaving_trjn

