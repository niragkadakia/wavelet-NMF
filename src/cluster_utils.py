"""
Utilities for clustering.

Created by Adam Fine and Nirag Kadakia at 11:07 09-12-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import numpy as np
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path


class lasso_select(object):
	"""
	Select indices from a matplotlib collection using `LassoSelector`.

	Selected indices are saved in the `ind` attribute. This tool fades out the
	points that are not part of the selection (i.e., reduces their alpha
	values). If your collection has alpha < 1, this tool will permanently
	alter the alpha values.

	Note that this tool selects collection objects based on their *origins*
	(i.e., `offsets`).

	Parameters
	----------
	ax: :class:`~matplotlib.axes.Axes`
		Axes to interact with.

	collection: :class:`matplotlib.collections.Collection` subclass
		Collection you want to select from.

	update_func: function
		Called after each new selection, argument is list of selected indices

	alpha_other: 0 <= float <= 1
		To highlight a selection, this tool sets all selected points to an
		alpha value of 1 and non-selected points to `alpha_other`.
	"""

	def __init__(self, ax, collection, update_func, alpha_other=0.3):
		
		self.canvas = ax.figure.canvas
		self.collection = collection
		self.alpha_other = alpha_other

		self.xys = collection.get_offsets()
		self.Npts = len(self.xys)

		# Ensure that we have separate colors for each object
		self.fc = collection.get_facecolors()
		if len(self.fc) == 0:
			raise ValueError('Collection must have a facecolor')
		elif len(self.fc) == 1:
			self.fc = np.tile(self.fc, (self.Npts, 1))

		self.lasso = LassoSelector(ax, onselect=self.onselect)
		self.update_func = update_func
		self.ind = []
		
	def update_vals(self):
		
		self.update_func(self.ind)

	def onselect(self, verts):
		
		path = Path(verts)
		self.ind = np.nonzero(path.contains_points(self.xys))[0]
		self.fc[:, -1] = self.alpha_other
		self.fc[self.ind, -1] = 1
		self.collection.set_facecolors(self.fc)
		self.canvas.draw_idle()
		
		# Function to call after new selection, based on new indices 
		self.update_vals()
		
	def disconnect(self):
		
		self.lasso.disconnect_events()
		self.fc[:, -1] = 1
		self.collection.set_facecolors(self.fc)
		self.canvas.draw_idle()


class hover_select():
	"""
	Get indices of single points over which the mouse is hovering.
	"""
	
	def __init__(self, fig, ax, pts, update_func):
		
		self.ax = ax
		self.pts = pts
		self.fig = fig
		self.update_func = update_func
		
	def update_vals(self):
		"""
		Just update with first index in region. Only plot one W.
		"""
		
		self.update_func(self.ind['ind'][0])
		
	def hover(self, event):
    
		if event.inaxes == self.ax:
			cont, self.ind = self.pts.contains(event)
			if cont:
				self.update_vals()
				self.fig.canvas.draw_idle()
			
