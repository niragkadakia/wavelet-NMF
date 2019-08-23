import numpy as np

def compute_max_corr_full(norm_behavior_1, norm_behavior_2):
	abs_max = 0
	real_max = 0
	max_index = -norm_behavior_1.shape[0]
	for delta in np.arange(1,norm_behavior_1.shape[0]):
		temp_val_minus = np.dot(norm_behavior_1[-delta:],norm_behavior_2[:delta])/delta
		if np.abs(temp_val_minus)>abs_max:
			abs_max = np.abs(temp_val_minus)
			real_max = temp_val_minus
			max_index = -(norm_behavior_1.shape[0]-delta)
	middle = np.dot(norm_behavior_1,norm_behavior_2)/norm_behavior_1.shape[0]
	if np.abs(middle)>abs_max:
		abs_max = np.abs(middle)
		real_max = middle
		max_index = 0
	for delta in np.arange(1,norm_behavior_1.shape[0]):
		temp_val_plus = np.dot(norm_behavior_1[:delta],norm_behavior_2[-delta:])/delta
		if np.abs(temp_val_plus)>abs_max:
			abs_max = np.abs(temp_val_plus)
			real_max = temp_val_plus
			max_index = norm_behavior_1.shape[0]-delta
	return real_max, abs_max, max_index

def compute_max_corr_range(norm_behavior_1, norm_behavior_2, min_offset, max_offset):
	abs_max = 0
	real_max = 0
	max_index = min_offset
	for delta in np.arange(min_offset, max_offset+1):
		temp_val = np.dot(norm_behavior_1[delta:],norm_behavior_2[:-delta])/norm_behavior_1[delta:].shape[0]
		if np.abs(temp_val)>abs_max:
			abs_max = np.abs(temp_val)
			real_max = temp_val
			max_index = delta
	return real_max, abs_max, max_index