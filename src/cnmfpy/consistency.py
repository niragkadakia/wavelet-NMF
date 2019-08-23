import numpy as np
from cnmf import CNMF
from conv import ShiftMatrix, shiftVector, vector_conv, tensor_conv
from scipy.stats.stats import pearsonr
from misc import computeNumFactors

# TODO: convert this to numpy
# computes the consistency between the factorizations in model 1 and model 2 as outlined in Mackevicius et al.
def compute2ModelConsistency(model_1, model_2):
	num = 0
	model_1_vector = []
	#piecewise_1 = []
	model_2_vector = []
	#piecewise_2 = []
	nf_1 = computeNumFactors(model_1)
	nf_2 = computeNumFactors(model_2)
	#print(nf_1, nf_2)
	if model_1.W.shape[2] != model_2.W.shape[2]:
		raise ValueError("Models have different parameters!")
	for nf_i in range(nf_1):
		reconstructed_part_1 = vector_conv(model_1.W[:,:,nf_i],shiftVector(model_1.H.shift(0)[nf_i],model_1.H.L),model_1._shifts)
		model_1_vector.append(reconstructed_part_1.flatten())
		#piecewise_1.append(reconstructed_part_1)

	#dif = checkConvolution(model_1,piecewise_1)
	#print(np.linalg.norm(dif))
	for nf_j in range(nf_2):
		reconstructed_part_2 = vector_conv(model_2.W[:, :, nf_j], shiftVector(model_2.H.shift(0)[nf_j], model_2.H.L),
		                                   model_2._shifts)
		model_2_vector.append(reconstructed_part_2.flatten())
	#print(model_1_vector)
	corr_matrix = getCorrMatrix(model_1_vector, model_2_vector)
	corr_matrix = reorderCorrMatrix(corr_matrix)
	for i in range(min(corr_matrix.shape[0],corr_matrix.shape[1])):
		num += corr_matrix[i,i]**2
	return num/np.linalg.norm(corr_matrix)**2

def compute1ModelConsistency(model, Wreal, Hreal):
	num = 0
	model_1_vector = []
	real_vector = []
	#piecewise_1 = []
	nf_1 = computeNumFactors(model)
	for nf_i in range(nf_1):
		reconstructed_part_1 = vector_conv(model.W[:,:,nf_i],shiftVector(model.H.shift(0)[nf_i],model.H.L),model._shifts)
		model_1_vector.append(reconstructed_part_1.flatten())
	for nf_j in range(len(Wreal)):
		reconstructed_part_2 = vector_conv(Wreal[nf_j], shiftVector(Hreal[nf_j], model.H.L), model._shifts)
		real_vector.append(reconstructed_part_2.flatten())
	#print(model_1_vector)
	corr_matrix = getCorrMatrix(model_1_vector, real_vector)
	corr_matrix_reordered = reorderCorrMatrix(np.copy(corr_matrix))
	for i in range(min(corr_matrix_reordered.shape[0],corr_matrix_reordered.shape[1])):
		num += corr_matrix_reordered[i,i]**2
	return num/np.linalg.norm(corr_matrix_reordered)**2, corr_matrix

def getCorrMatrix(v1, v2):
	corr_matrix = np.zeros((len(v1),len(v2)))
	for i in range(len(v1)):
		for j in range(len(v2)):
			corr_matrix[i, j] = pearsonr(v1[i], v2[j])[0]
	return corr_matrix

def reorderCorrMatrix(matrix):
	for i in range(min(matrix.shape[0],matrix.shape[1])):
		index = np.unravel_index(np.argmax(matrix[i:,i:], axis=None), matrix[i:,i:].shape)
		matrix[[i,index[0]+i],:] = matrix[[index[0]+i,i],:]
		matrix[:,[i, index[1]+i]] = matrix[:, [index[1]+i,i]]
	return matrix

def checkConvolution(model, piecewise):
	model_predict = model.predict()
	sum_conv = np.zeros((model_predict.shape))
	for i in range(len(piecewise)):
		sum_conv += piecewise[i]
	return sum_conv-model_predict