import numpy as np
import matplotlib.pyplot as plt
import landweberHelper as lh


def landweber(param_struct, c_m_min, c_m_max, c_measure, norm_struct, s_mat, a_lw, iter_i):
    c = lh.cap_norm(param_struct, c_m_min, c_m_max, c_measure, norm_struct)
    g = np.dot(s_mat.T, c)
    # N = g.size[0]
    # Z = np.zeros(1, param_struct.nlist.size[0])
    if anim == 1:
        for ii in range(1, iter_i):
            g = g + a_lw * np.dot(s_mat.T, (c-np.dot(s_mat, g)))
            g[g < 0] = 0
            g[g >= 1] = 1
    elif anim == 0:
        for ii in range(1, iter_i):
            g = g + a_lw * np.dot(s_mat.T, (c-np.dot(s_mat, g)))
            g[g < 0] = 0
            g[g >= 1] = 1
    return g


datapath = '/home/andre/Documents/Git/TPA_pub/Daten/'
param = lh.read_matlab_struct(datapath + 'param.mat')['param']
norm = param.norm
anim = param.anim

cap_min = np.array(lh.read_cap_file(datapath + 'c_min_ij.txt', param))
cap_max = np.array(lh.read_cap_file(datapath + 'c_max_ij.txt', param))
c_array = np.array(lh.read_cap_file(datapath + 'Herz.txt', param))
s = lh.read_matlab_var(datapath + 'S_matrix_q_neu_richtig.mat')

lw_iter = 20
iterations = 100

solved = landweber(param, cap_min, cap_max, c_array, norm, s, lw_iter, iterations)
solved_image = np.reshape(solved, (91,91))
plt.imshow(np.log(solved_image))
plt.show()