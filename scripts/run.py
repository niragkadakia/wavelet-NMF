import sys
sys.path.append("../src")
sys.path.append("../src/cnmfpy")
from beh_NMF import *

idx = int(sys.argv[1])

#a = wavelet_transform()
#a.transform()
a = NMF(seqnmf_norm_idx=idx)
a.factorize()
