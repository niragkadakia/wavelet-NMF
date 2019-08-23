import numpy as np
from conv import vector_conv, shiftVector
import pickle
import os
try:
	import winsound
except ImportError:
	pass

def computeIndices(num):
	i = 1
	while i*(i+1)/2 < num:
		i += 1
	return i, int(num-i*(i-1)/2)

def computeNumFactors(model):
	factor = 0
	for factor in range(model.W.shape[2]):
		if not model.W[:, :, factor].any():
			return factor
	return model.W.shape[2]

def computeGroundTruthMatrix(factor_list, timetrace_list, maxlag):
	result = np.zeros((factor_list[0].shape[1],timetrace_list[0].shape[0]))
	for i in range(len(factor_list)):
		result += vector_conv(factor_list[i],shiftVector(
					timetrace_list[i],maxlag),np.arange(maxlag * 2 + 1) - maxlag)
	return result

def getDataDir():
	return "../../../../data"

def rgetDataDir():
	return "../../../../data/real"


def getDataParams():
	fpath = os.path.join(getDataDir(),"data_params.pkl")
	with open(fpath, "rb") as f:
		data_params = pickle.load(f)
	return data_params

def rgetDataParams(subdir):
	fpath = os.path.join(rgetDataDir(),subdir+"/data_params.pkl")
	with open(fpath, "rb") as f:
		data_params = pickle.load(f)
	return data_params

def getPickledData():
	fpath = os.path.join(getDataDir(),"data.pkl")
	with open(fpath, "rb") as f:
		temp_data = pickle.load(f)
	return temp_data

def rgetPickledData(subdir):
	fpath = os.path.join(rgetDataDir(),subdir+"/data.pkl")
	with open(fpath, "rb") as f:
		temp_data = pickle.load(f)
	return temp_data


def lowToZero(matrix, tol):
	smoothed_matrix = matrix.copy()
	smoothed_matrix[smoothed_matrix<tol] = 0
	return smoothed_matrix

def getNonzeroSlice(matrix):
	initial = 0
	final = matrix.shape[0]
	for i in range(matrix.shape[0]):
		initial = i
		if matrix[initial,:].any():
			break
	for j in range(matrix.shape[0]):
		final = matrix.shape[0] - j
		if matrix[final-1,:].any():
			break
	return initial,final

def computeDensity(matrix, tol):
	smatrix = lowToZero(matrix, tol)
	if smatrix.any():
		initial, final = getNonzeroSlice(smatrix)
		density = np.count_nonzero(smatrix[initial:final,:])/smatrix[initial:final,:].size
	else:
		density = 0
	return density

def value_in_interior(peak_indices, cidx_ranges, half_width=30):
	idx_list = []
	for crange in cidx_ranges:
		sorted_idxs = np.searchsorted(crange, peak_indices)
		sorted_idxs = np.minimum(sorted_idxs, crange.shape[0]-sorted_idxs)
		idx_list.append(sorted_idxs)
		#idx_loc = np.concatenate((idx_loc, sorted_idxs), axis=0)
	min_idx_matrix = np.amax(np.asarray(idx_list), axis=0)
	return np.asarray(peak_indices)[min_idx_matrix > half_width]

def quit_w():
	winsound.Beep(440, 2000)
	quit()

def conv_ctu_or_utc(indices, cidx_ranges, uidx_ranges, mode="utc"):
	uidx_ranges_1d = np.hstack(uidx_ranges)
	cidx_ranges_1d = np.hstack(cidx_ranges)
	if mode == "utc":
		if np.all(np.isin(indices,uidx_ranges_1d)):
			return cidx_ranges_1d[np.searchsorted(uidx_ranges_1d,indices)]
		else:
			raise ValueError("One or more indices is out of range!")

	elif mode == "ctu":
		if np.all(np.isin(indices, cidx_ranges_1d)):
			return uidx_ranges_1d[np.searchsorted(cidx_ranges_1d,indices)]
		else:
			raise ValueError("One or more indices is out of range!")
	else:
		raise ValueError("invalid mode")