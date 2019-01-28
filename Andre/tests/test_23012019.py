import numpy as np
import landweberHelper as lH
import matplotlib.pyplot as plt
import os

phantomdir = 'streifenx'
datapath = '/home/andre/Documents/Studium/Teamprojektarbeit/Datenundshit/Daten_TPA_23012019/TPA_Daten_fixed.mat'
phantompath = '/home/andre/Documents/Studium/Teamprojektarbeit/Matlab Toolbox/Matlab/' + phantomdir + '/c_phantom4.txt'
minpath = '/home/andre/Documents/Studium/Teamprojektarbeit/Matlab Toolbox/Matlab/' + phantomdir + '/c_min_ij.txt'
maxpath = '/home/andre/Documents/Studium/Teamprojektarbeit/Matlab Toolbox/Matlab/' + phantomdir + '/c_max_ij.txt'


import_struct = lH.read_matlab_struct(datapath)

param = import_struct['param']
cmats = import_struct['cmats']
S = import_struct['solu'].S_r

import_phantom = lH.read_cap_file(phantompath, param)
import_phantom_min = lH.read_cap_file(minpath, param)
import_phantom_max = lH.read_cap_file(maxpath, param)

#solution = lH.landweber(param, cmats.C_m_min, cmats.C_m_max, cmats.C_phantom, param.norm, S, 5, 5000)
solution = lH.landweber(param, import_phantom_min, import_phantom_max, import_phantom, param.norm, S, 5, 800)
lH.plot_ect(solution)


# c_min = cmats.C_m_min
# c_max = cmats.C_m_max
# c_n = cmats.C_phantom
#
# C_norm = np.divide((c_n - c_min),(c_max - c_min))
# g = np.dot(S.T, C_norm)
# N = g.shape[0]
# a_lw = 5
#
# for ii in range(0, 5000):
#     g = g + a_lw*np.dot(S.T, (C_norm - np.dot(S,g)))
#     g[g < 0] = 0
#     g[g >= 1] = 1
#     print(ii)
#
# plt.imshow(np.reshape(g, (91, 91)))
# plt.show()
# plt.clim(0, 0.2)