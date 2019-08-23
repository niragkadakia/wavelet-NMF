"""
TODO

Created by Nirag Kadakia at 20:30 04-22-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys

sys.path.append("../src")
sys.path.append("../src/cnmfpy")

"""
import os
import pickle
import scipy as sp
import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.ticker as mticker
from cwt_utils import get_cwt_scales
from cnmf import CNMF, ShiftMatrix
from regularize import compute_scfo_reg
from misc import computeNumFactors
from plotting import plotCWTNMF
"""

from beh_NMF import *
#a = wavelet_transform()
#a.transform()
a = NMF(seqnmf_norm_idx=0)
a.factorize()
a.plot()