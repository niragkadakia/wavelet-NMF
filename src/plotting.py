import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from misc import computeNumFactors, value_in_interior, rgetDataDir, getDataDir
import numpy as np
from conv import shiftVector, vector_conv
import peakutils
import os

subdir = "Ribbon"


def plotData(data, Wreal, Hreal):
	total_time = Hreal[0].shape[0]
	widths = [5, 7]
	for i in range(len(Wreal)):
		widths.append(2)

	fig = plt.figure(figsize=(17, 9))
	gs = gridspec.GridSpec(len(Hreal), 2 + len(Wreal), width_ratios=widths)
	for i in range(len(Hreal)):
		ax = plt.subplot(gs[i, 0])
		if i != len(Hreal) - 1:
			ax.xaxis.set_visible(False)
		plt.plot(range(total_time), Hreal[i])
		ax = None
	plt.subplot(gs[:, 1])
	plt.title("Real")
	plt.imshow(data)

	for i in range(len(Wreal)):
		ax = plt.subplot(gs[:, i + 2])
		if i != 0:
			ax.yaxis.set_visible(False)
		ax = None
		plt.imshow(Wreal[i].T)
	return fig


def plotModel(model, num_real_components, data, Hreal, Wreal=None, title=""):
	num_estimated_components = model.H.shape[0]
	total_time = model.H.shape[1]
	num_columns = 2
	widths = [5, 10]
	if Wreal:
		for i in range(len(Wreal)):
			widths.append(2)
		num_columns += len(Wreal)
		real_columns = len(Wreal)
	else:
		real_columns = 0
	max_factor = computeNumFactors(model)
	num_columns += max_factor
	for i in range(max_factor):
		widths.append(2)

	fig = plt.figure(figsize=(17, 9))
	plt.suptitle(title)
	gs = gridspec.GridSpec(num_real_components + num_estimated_components, num_columns, width_ratios=widths)
	gs.update(left=.04, wspace=.05, hspace=.05)
	for i in range(num_real_components):
		ax = plt.subplot(gs[i, 0])
		ax.xaxis.set_visible(False)
		plt.plot(range(total_time), Hreal[i])
		ax = None
	for i in range(num_real_components + 1, num_real_components + num_estimated_components + 1):
		ax = plt.subplot(gs[i - 1, 0])
		if i != num_real_components + num_estimated_components:
			ax.xaxis.set_visible(False)
		ax = None
		plt.plot(range(total_time), model.H.shift(0)[i - 1 - num_real_components], c="y")

	plt.subplot(gs[:(num_real_components + num_estimated_components) // 2, 1])
	plt.title("Real")
	plt.imshow(data)
	plt.subplot(gs[(num_real_components + num_estimated_components + 1) // 2:, 1])
	plt.title("Predicted")
	plt.imshow(model.predict())
	for j in range(real_columns):
		ax = plt.subplot(gs[:, j + 2])
		if j != 0:
			ax.yaxis.set_visible(False)
		plt.imshow(Wreal[j].T)
		ax = None
	for i in range(max_factor):
		ax = plt.subplot(gs[:, i + real_columns + 2])
		if i != 0 or real_columns != 0:
			ax.yaxis.set_visible(False)
		plt.imshow(model.W[:, :, i].T)
		ax = None

	return fig


def plotGTModel(model, num_real_components, data, Hreal, Wreal=None, title=""):
	total_time = model.H.shape[1]
	num_factors = computeNumFactors(model)
	print(num_factors)
	num_rows = num_real_components + model.W.shape[2]

	fig = plt.figure(figsize=(17, 9))
	plt.suptitle(title)
	gs = gridspec.GridSpec(1, 3, width_ratios=[1,4,3])
	gs.update(left=.04, wspace=.05, hspace=.01)
	h_gs = gridspec.GridSpecFromSubplotSpec(num_rows, 1, gs[0])
	for i in range(num_real_components):
		ax = plt.subplot(h_gs[i])
		ax.yaxis.set_visible(False)
		ax.xaxis.set_visible(False)
		ax.plot(range(total_time), Hreal[i])
	for i in range(model.W.shape[2]):
		ax = plt.subplot(h_gs[i + num_real_components])
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		ax.plot(range(total_time), model.H.shift(0)[i], c="y")


	m_gs = gridspec.GridSpecFromSubplotSpec(2,1,gs[1])
	ax = plt.subplot(m_gs[0])
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	plt.title("Real")
	ax.imshow(data, aspect="auto")
	ax = plt.subplot(m_gs[1])
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	plt.title("Predicted")
	ax.imshow(model.predict(), aspect="auto")

	factor_gs = gridspec.GridSpecFromSubplotSpec(3,num_real_components,gs[2])
	for j in range(num_real_components):
		ax = plt.subplot(factor_gs[0,j])
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		ax.imshow(Wreal[j].T, aspect="auto")
	for i in np.arange(model.W.shape[2]):
		ax = plt.subplot(factor_gs[i//num_real_components+1,i%num_real_components])
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		ax.imshow(model.W[:, :, i].T, aspect="auto")

	return fig


def plotRealModel(model, data, title=""):
	total_time = model.H.shape[1]
	num_factors = computeNumFactors(model)
	print(num_factors)
	num_rows = num_factors
	widths = [7, 2, 2, 5]

	fig = plt.figure(figsize=(17, 9))
	plt.suptitle(title)
	gs = gridspec.GridSpec(num_rows, 4, width_ratios=widths)
	gs.update(left=.04, wspace=.05, hspace=.05)
	for i in range(num_factors):
		ax = plt.subplot(gs[i, 0])
		plt.plot(range(total_time), model.H.shift(0)[i], c="y")
		ax.xaxis.set_visible(False)

	plt.subplot(gs[:, 1])
	plt.title("Real")
	plt.imshow(data.T, aspect="auto")
	plt.subplot(gs[:, 2])
	plt.title("Predicted")
	plt.imshow(model.predict().T, aspect="auto")

	for i in range(num_factors):
		ax = plt.subplot(gs[i, 3])
		plt.imshow(model.W[:, :, i].T)
		ax.xaxis.set_visible(False)

	return fig


def plotPCAComponents(w_list, max_indices, num_to_plot, num_cols=20):
	w_zeros = np.zeros(w_list[0].shape)
	fig_list = []
	for num in range(num_to_plot):
		num_rows = int(np.ceil(len(max_indices[num]) / num_cols))
		fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 3 * num_rows))
		fig.suptitle("Component {}".format(num))
		fig.dpi = 500
		for row in range(num_rows - 1):
			for obj in range(num_cols):
				axes[row, obj].yaxis.set_visible(False)
				axes[row, obj].xaxis.set_visible(False)
				axes[row, obj].imshow(w_list[max_indices[num][obj + row * num_cols]].T)
		row = num_rows - 1
		for obj in range(row * num_cols, len(max_indices[num])):
			axes[row, obj % num_cols].yaxis.set_visible(False)
			axes[row, obj % num_cols].xaxis.set_visible(False)
			axes[row, obj % num_cols].imshow(w_list[max_indices[num][obj]].T)
		for obj in range(len(max_indices[num]), num_rows * num_cols):
			axes[row, obj % num_cols].yaxis.set_visible(False)
			axes[row, obj % num_cols].xaxis.set_visible(False)
			axes[row, obj % num_cols].imshow(w_zeros.T)
		fig_list.append(fig)
	return fig_list


def plotCWTNMF(model, cwt_data, raw_data, data_params, title=""):
	total_time = model.H.shape[1]
	num_factors = computeNumFactors(model)
	print(num_factors)
	# num_rows = max(num_factors, 2)
	# num_cols = 2 + int(np.ceil(num_factors / 3))
	widths = [5, 20, 10]
	# for ex_col in np.arange(2, num_cols):
	# widths.append(2)

	fig_1 = plt.figure()
	plt.suptitle(title)
	gs = gridspec.GridSpec(2, 3, width_ratios=widths)
	gs.update(left=0.03, right=.97, wspace=.05, hspace=.02)
	left_gs = gridspec.GridSpecFromSubplotSpec(num_factors, 1, gs[:, 0])
	for i in np.arange(num_factors):
		ax = plt.subplot(left_gs[i])
		ax.plot(np.arange(total_time), model.H.shift(0)[i], c="y")
		ax.xaxis.set_visible(False)

	inner_gridspec_1 = gridspec.GridSpecFromSubplotSpec(data_params["num_vars"], 1, gs[0, 1], hspace=0)
	xr = np.arange(cwt_data.shape[1]) / 19
	freqs = np.geomspace(data_params["upper_freq"], data_params["lower_freq"], data_params["num_neurons_per_var"])
	fake_data = model.predict()
	vmin = min(np.amin(cwt_data), np.amin(fake_data))
	vmax = max(np.amax(cwt_data), np.amax(fake_data))
	axes_list = []
	first_inner_ax = plt.subplot(inner_gridspec_1[0])
	plt.title("Real")
	axes_list.append(first_inner_ax)
	for i in np.arange(1, data_params["num_vars"]):
		ax_temp = plt.subplot(inner_gridspec_1[i], sharex=first_inner_ax, sharey=first_inner_ax)
		axes_list.append(ax_temp)
	for num, axes in enumerate(axes_list):
		axes.pcolormesh(xr, freqs, cwt_data[num * data_params["num_neurons_per_var"]:(num + 1) * data_params[
			"num_neurons_per_var"], :], vmin=vmin, vmax=vmax)
		axes.set_yscale("log")
		axes.set_xlim(0, cwt_data.shape[1] / 19)
		axes.yaxis.set_ticks(np.arange(np.ceil(freqs[-1] * 2) / 2, freqs[0], .5))
		axes.yaxis.set_major_formatter(mticker.ScalarFormatter())

	inner_gridspec_2 = gridspec.GridSpecFromSubplotSpec(data_params["num_vars"], 1, gs[1, 1], hspace=0)

	for j in np.arange(data_params["num_vars"]):
		ax_temp = plt.subplot(inner_gridspec_2[j], sharex=first_inner_ax, sharey=first_inner_ax)
		ax_temp.pcolormesh(xr, freqs, fake_data[j * data_params["num_neurons_per_var"]:(j + 1) * data_params[
			"num_neurons_per_var"], :], vmin=vmin, vmax=vmax)
		ax_temp.set_yscale("log")
		ax_temp.set_xlim(0, cwt_data.shape[1] / 19)
		ax_temp.yaxis.set_ticks(np.arange(np.ceil(freqs[-1]) / 2, freqs[0], .5))
		ax_temp.yaxis.set_major_formatter(mticker.ScalarFormatter())
		if j == 0:
			plt.title("Predicted")

	num_cols = int(np.ceil(num_factors / 3))
	right_gs = gridspec.GridSpecFromSubplotSpec(3, num_cols, gs[:, 2], hspace=.01, wspace=.05)
	if num_factors > 3:
		for i in np.arange(num_factors):
			b = i % num_cols
			a = i // num_cols
			ax = plt.subplot(right_gs[a, b])
			ax.imshow(model.W[:, :, i].T, aspect="auto")
			for extra_var in np.arange(1, data_params["num_vars"]):
				ax.plot(np.arange(model.W[:, :, i].T.shape[1]),
				        np.zeros((model.W[:, :, i].T.shape[1])) + extra_var * data_params["num_neurons_per_var"], "r",
				        linewidth=.5)
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
	else:
		for i in range(num_factors):
			ax = plt.subplot(right_gs[i])
			ax.imshow(model.W[:, :, i].T, aspect="auto")
			for extra_var in np.arange(1, data_params["num_vars"]):
				ax.plot(np.arange(model.W[:, :, i].T.shape[1]),
				        np.zeros((model.W[:, :, i].T.shape[1])) + extra_var * data_params["num_neurons_per_var"], "r",
				        linewidth=.5)
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
	return fig_1


def plot_arb_components(model, comp_list, cwt_data, raw_data, data_params):
	main_fig = plt.figure()
	plt.suptitle("h matrices: {0}".format(comp_list))
	xr = np.arange(cwt_data.shape[1]) / 19
	freqs = np.geomspace(data_params["upper_freq"], data_params["lower_freq"], data_params["num_neurons_per_var"])

	pattern = np.zeros((model.W.shape[1], model.H.shape[1]))
	for component in comp_list:
		pattern = pattern + vector_conv(model.W[:, :, component], shiftVector(model.H.shift(0)[component], model.H.L),
		                                model._shifts)
	vmin = min(np.amin(cwt_data), np.amin(pattern))
	vmax = max(np.amax(cwt_data), np.amax(pattern))

	outer_gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3, 3])
	raw_data_gs = gridspec.GridSpecFromSubplotSpec(data_params["num_vars"], 1, outer_gs[0])
	first_top_ax = plt.subplot(raw_data_gs[0])
	first_top_ax.plot(raw_data[0, :])
	first_top_ax.set_xlim(0, xr[-1])
	for ex_var in np.arange(1, data_params["num_vars"]):
		temp_top_ax = plt.subplot(raw_data_gs[ex_var], sharex=first_top_ax)
		temp_top_ax.plot(raw_data[1, :])

	real_data_gs = gridspec.GridSpecFromSubplotSpec(data_params["num_vars"], 1, outer_gs[1], hspace=0)
	first_real_ax = plt.subplot(real_data_gs[0], sharex=first_top_ax)
	plt.title("Real")
	axes_list = []
	axes_list.append(first_real_ax)
	for i in np.arange(1, data_params["num_vars"]):
		ax_temp = plt.subplot(real_data_gs[i], sharex=first_top_ax, sharey=first_real_ax)
		axes_list.append(ax_temp)
	for num, axes in enumerate(axes_list):
		axes.pcolormesh(xr, freqs, cwt_data[num * data_params["num_neurons_per_var"]:(num + 1) * data_params[
			"num_neurons_per_var"], :], vmin=vmin, vmax=vmax)
		axes.set_yscale("log")
		axes.set_xlim(0, xr[-1])
		axes.yaxis.set_ticks(np.arange(np.ceil(freqs[-1] * 2) / 2, freqs[0], .5))
		axes.yaxis.set_major_formatter(mticker.ScalarFormatter())

	patterns_gs = gridspec.GridSpecFromSubplotSpec(data_params["num_vars"], 1, outer_gs[2], hspace=0)
	first_pattern_ax = plt.subplot(patterns_gs[0], sharex=first_top_ax)
	plt.title("Matrices {0}".format(comp_list))
	axes_list = []
	axes_list.append(first_pattern_ax)
	for i in np.arange(1, data_params["num_vars"]):
		ax_temp = plt.subplot(patterns_gs[i], sharex=first_top_ax, sharey=first_pattern_ax)
		axes_list.append(ax_temp)
	for num, axes in enumerate(axes_list):
		axes.pcolormesh(xr, freqs, pattern[num * data_params["num_neurons_per_var"]:(num + 1) * data_params[
			"num_neurons_per_var"], :], vmin=vmin, vmax=vmax)
		axes.set_yscale("log")
		axes.set_xlim(0, xr[-1])
		axes.yaxis.set_ticks(np.arange(np.ceil(freqs[-1] * 2) / 2, freqs[0], .5))
		axes.yaxis.set_major_formatter(mticker.ScalarFormatter())
	return main_fig


def plot_h_peaks(model, raw_data, uidx_ranges, cidx_ranges, exp_matrix_dict, h_set=None, half_width=20, max_peaks=10):
	uidx_ranges_conc = np.concatenate(uidx_ranges, axis=0)
	h_set_indexes = []
	if h_set is None:
		h_set = [0]
	for h_index in h_set:
		base_thresh = .1
		peak_indexes = peakutils.indexes(model.H.shift(0)[h_index], thres=base_thresh, min_dist=half_width)
		good_peak_indexes = value_in_interior(peak_indexes, cidx_ranges, half_width=half_width)
		while len(good_peak_indexes) > max_peaks:
			base_thresh += .02
			peak_indexes = peakutils.indexes(model.H.shift(0)[h_index], thres=base_thresh, min_dist=half_width)
			good_peak_indexes = value_in_interior(peak_indexes, cidx_ranges, half_width=half_width)
		fig_temp = plt.figure()
		fig_temp.suptitle("w matrix {}".format(h_index))
		gs = gridspec.GridSpec(4, (len(good_peak_indexes) + 1) // 2, height_ratios=[1, 4, 1, 4])
		for i, good_index in enumerate(good_peak_indexes):
			ax1 = plt.subplot(gs[2 * (i % 2), i // 2])
			ax1.scatter(np.arange(good_index - half_width, good_index + half_width) / 19,
			            raw_data[good_index - half_width:good_index + half_width],
			            c=model.H.shift(0)[h_index][good_index - half_width:good_index + half_width],
			            vmin=np.amin(model.H.shift(0)[h_index]),
			            vmax=np.amax(model.H.shift(0)[h_index][good_peak_indexes]))
			ax2 = plt.subplot(gs[2 * (i % 2) + 1, i // 2])
			ax2.plot(exp_matrix_dict["x"][uidx_ranges_conc[good_index - half_width:good_index + half_width]],
			         exp_matrix_dict["y"][uidx_ranges_conc[good_index - half_width:good_index + half_width]], lw=.5,
			         color="k")
			ax2.scatter(exp_matrix_dict["x"][uidx_ranges_conc[good_index - half_width:good_index + half_width]],
			            exp_matrix_dict["y"][uidx_ranges_conc[good_index - half_width:good_index + half_width]],
			            c=model.H.shift(0)[h_index][good_index - half_width:good_index + half_width], marker=".",
			            vmin=np.amin(model.H.shift(0)[h_index]),
			            vmax=np.amax(model.H.shift(0)[h_index][good_peak_indexes]))
			start_x = exp_matrix_dict["x"][uidx_ranges_conc[good_index - half_width]]
			end_x = exp_matrix_dict["x"][uidx_ranges_conc[good_index + half_width - 1]]
			start_y = exp_matrix_dict["y"][uidx_ranges_conc[good_index - half_width]]
			end_y = exp_matrix_dict["y"][uidx_ranges_conc[good_index + half_width - 1]]
			total_length = np.sqrt((.6 * (end_x - start_x)) ** 2 + (.6 * (end_y - start_y) ** 2))
			ax2.arrow(.8 * start_x + .2 * end_x, .8 * start_y + .2 * end_y, .6 * (end_x - start_x),
			          .6 * (end_y - start_y), length_includes_head=True, color="k", alpha=.2,
			          head_width=.05 * total_length, overhang=.9)
			plt.axis("equal")
			figManager = plt.get_current_fig_manager()
			figManager.window.state("zoomed")


def plot_trajs_nirag_special(h, raw_data, uidx_ranges, cidx_ranges, exp_matrix_dict, half_width=20):
	uidx_ranges_conc = np.concatenate(uidx_ranges, axis=0)
	base_thresh = .1
	peak_indices = peakutils.indexes(h, thres=base_thresh, min_dist=half_width)
	good_peak_indexes = value_in_interior(peak_indices, cidx_ranges, half_width=half_width)
	while len(good_peak_indexes) > 10:
		base_thresh += .02
		peak_indices = peakutils.indexes(h, thres=base_thresh, min_dist=half_width)
		good_peak_indexes = value_in_interior(peak_indices, cidx_ranges, half_width=half_width)
	good_sublist = []
	good_omega = []
	for good_index in good_peak_indexes:
		good_sublist.append(raw_data[good_index - half_width:good_index + half_width])
		good_sublist.append(exp_matrix_dict["x"][uidx_ranges_conc[good_index - half_width:good_index + half_width]])
		good_sublist.append(exp_matrix_dict["y"][uidx_ranges_conc[good_index - half_width:good_index + half_width]])
	fig_temp = plt.figure()
	gs = gridspec.GridSpec(4, (len(good_peak_indexes) + 1) // 2, height_ratios=[1, 4, 1, 4])
	for i, good_index in enumerate(good_peak_indexes):
		ax1 = plt.subplot(gs[2 * (i % 2), i // 2])
		ax1.scatter(np.arange(good_index - half_width, good_index + half_width) / 19,
		            raw_data[good_index - half_width:good_index + half_width],
		            c=h[good_index - half_width:good_index + half_width], vmin=np.amin(h), vmax=np.amax(h))
		ax2 = plt.subplot(gs[2 * (i % 2) + 1, i // 2])
		ax2.plot(exp_matrix_dict["x"][uidx_ranges_conc[good_index - half_width:good_index + half_width]],
		         exp_matrix_dict["y"][uidx_ranges_conc[good_index - half_width:good_index + half_width]], lw=.5,
		         color="k")
		ax2.scatter(exp_matrix_dict["x"][uidx_ranges_conc[good_index - half_width:good_index + half_width]],
		            exp_matrix_dict["y"][uidx_ranges_conc[good_index - half_width:good_index + half_width]],
		            c=h[good_index - half_width:good_index + half_width], marker=".",
		            vmin=np.amin(h), vmax=np.amax(h))
		plt.axis("equal")
	figManager = plt.get_current_fig_manager()
	figManager.window.state("zoomed")
	return good_sublist



def save_data_single_h_traj(h, idx, raw_data, uidx_ranges, cidx_ranges, exp_matrix_dict, half_width=20, max_peaks=4):
	print(np.amin(h))
	return
# 	times_list = []
# 	w_list = []
# 	color_list = []
# 	x_list = []
# 	y_list = []
# 	uidx_ranges_conc = np.concatenate(uidx_ranges, axis=0)
# 	base_thresh = .1
# 	peak_indexes = peakutils.indexes(h, thres=base_thresh, min_dist=half_width)
# 	good_peak_indexes = value_in_interior(peak_indexes, cidx_ranges, half_width=half_width)
# 	while len(good_peak_indexes) > max_peaks:
# 		print(len(good_peak_indexes))
# 		base_thresh += .02
# 		peak_indexes = peakutils.indexes(h, thres=base_thresh, min_dist=half_width)
# 		good_peak_indexes = value_in_interior(peak_indexes, cidx_ranges, half_width=half_width)
# 	for i, good_index in enumerate(good_peak_indexes):
# 		times_list.append(np.arange(good_index - half_width, good_index + half_width) / 19))
# 		,
# 		            raw_data[good_index - half_width:good_index + half_width],
# 		            c=h[good_index - half_width:good_index + half_width],
# 		            vmin=np.amin(h),
# 		            vmax=np.amax(h[good_peak_indexes]))
# 		ax2 = plt.subplot(gs[2 * (i % 2) + 1, i // 2])
# 		ax2.plot(exp_matrix_dict["x"][uidx_ranges_conc[good_index - half_width:good_index + half_width]],
# 		         exp_matrix_dict["y"][uidx_ranges_conc[good_index - half_width:good_index + half_width]], lw=.5,
# 		         color="k")
# 		ax2.scatter(exp_matrix_dict["x"][uidx_ranges_conc[good_index - half_width:good_index + half_width]],
# 		            exp_matrix_dict["y"][uidx_ranges_conc[good_index - half_width:good_index + half_width]],
# 		            c=h[good_index - half_width:good_index + half_width], marker=".",
# 		            vmin=np.amin(h), vmax=np.amax(h[good_peak_indexes]))
# 	# plt.savefig(os.path.join(rgetDataDir(), subdir + "/temp/{0}.png".format(idx)))
# 	plt.show()
# # plt.savefig(os.path.join(getDataDir(), "patterns_last_{0}.png".format(idx)))
# # plt.close(fig_temp)

def plotCWTNMFBetter(model, cwt_data, data_params, num_factors_to_plot=5, title=""):
	total_time = model.H.shape[1]
	num_factors = computeNumFactors(model)
	print(num_factors)
	# num_rows = max(num_factors, 2)
	# num_cols = 2 + int(np.ceil(num_factors / 3))
	widths = [3, 10]
	# for ex_col in np.arange(2, num_cols):
	# widths.append(2)
	heights = []
	for i in np.arange(num_factors_to_plot):
		heights.append(2)
	heights.append(15)

	fig_1 = plt.figure()
	plt.suptitle(title)
	gs = gridspec.GridSpec(num_factors_to_plot+1, 2, width_ratios=widths, height_ratios=heights, hspace=.5, wspace=.01)
	gs.update(left=0.05, right=.97)
	left_gs = gridspec.GridSpecFromSubplotSpec(1, num_factors_to_plot, gs[-1, 0], hspace=.01)
	for i in np.arange(num_factors_to_plot):
		ax = plt.subplot(left_gs[i])
		ax.imshow(model.W[:, :, i].T, aspect="auto")
		for extra_var in np.arange(1, data_params["num_vars"]):
			ax.plot(np.arange(model.W[:, :, i].T.shape[1]),
			        np.zeros((model.W[:, :, i].T.shape[1])) + extra_var * data_params["num_neurons_per_var"],
			        "r",
			        linewidth=.5)
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		# ax = plt.subplot(left_gs[i])
		# ax.plot(np.arange(total_time), model.H.shift(0)[i], c="y")
		# ax.xaxis.set_visible(False)

	mid_top_gs = gridspec.GridSpecFromSubplotSpec(num_factors_to_plot, 1, gs[:-1, 1], hspace=.01)
	for i in np.arange(num_factors_to_plot):
		ax = plt.subplot(mid_top_gs[i])
		ax.plot(np.arange(total_time), model.H.shift(0)[i], c="b")
		ax.xaxis.set_visible(False)
		ax.set_xlim(0, cwt_data.shape[1])
	data_plt = gridspec.GridSpecFromSubplotSpec(data_params["num_vars"],1,gs[-1,1], hspace=0.0)
	xr = np.arange(cwt_data.shape[1]) / 19
	freqs = np.geomspace(data_params["upper_freq"], data_params["lower_freq"], data_params["num_neurons_per_var"])
	fake_data = model.predict()
	vmin = min(np.amin(cwt_data), np.amin(fake_data))
	vmax = max(np.amax(cwt_data), np.amax(fake_data))
	axes_list = []
	for i in np.arange(data_params["num_vars"]):
		ax_temp = plt.subplot(data_plt[i])
		axes_list.append(ax_temp)
		if i == 0:
			plt.title("Real")
			ax_temp.xaxis.set_visible(False)
	for num, axes in enumerate(axes_list):
		axes.pcolormesh(xr, freqs, cwt_data[num * data_params["num_neurons_per_var"]:(num + 1) * data_params[
			"num_neurons_per_var"], :], vmin=vmin, vmax=vmax)
		axes.set_yscale("log")


		axes.minorticks_off()
		axes.set_yticks([])
		# axes.yaxis.set_major_formatter(mticker.ScalarFormatter())
		# axes.yaxis.set_ticks(np.arange(np.ceil(freqs[-1] * 2) / 2, freqs[0], 4))


		axes.set_xlim(0, cwt_data.shape[1] / 19)

	return fig_1
