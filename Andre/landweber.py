import numpy as np
import matplotlib.pyplot as plt
import landweberHelper as lH


def landweber(param_struct, c_m_min, c_m_max, c_measure, norm_struct, s_mat, a_lw, iter_i):
    """Execute Landweber algorithm"""
    c = lH.cap_norm(param_struct, c_m_min, c_m_max, c_measure, norm_struct)
    g = np.dot(s_mat.T, c)
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
param = lH.read_matlab_struct(datapath + 'param.mat')['param']
norm = param.norm
anim = param.anim

cap_min = np.array(lH.read_cap_file(datapath + 'c_min_ij.txt', param))
cap_max = np.array(lH.read_cap_file(datapath + 'c_max_ij.txt', param))
c_array = np.array(lH.read_cap_file(datapath + 'Herz.txt', param))
s = lH.read_matlab_var(datapath + 'S_matrix_q_neu_richtig.mat')

lw_iter = 20
iterations = 100

solved = landweber(param, cap_min, cap_max, c_array, norm, s, lw_iter, iterations)
solved_image = np.reshape(solved, (91, 91))

fig1 = plt.figure()
plot1 = plt.imshow(np.log(solved_image))
plt.grid()
plt.show()
