import numpy as np
from LBP_helper import *


def cap_norm(param_struct, c_min, c_max, c, norm_struct):
    """ Normalize the capacity values"""
    if param_struct.ind_norm == 1:
        if norm_struct == 's':
            c_norm = np.zeros((param_struct.M, 1))
            for ii in range(1, param_struct.M):
                c_norm[ii] = (1/c[ii]-1/c_min[ii])/(1/c_max[ii]-1/c_min[ii])
        elif norm_struct == 'p':
            c_norm = np.zeros((param_struct.M, 1))
            for ii in range(1, param_struct.M):
                c_norm[ii] = (c[ii]-c_min[ii])/(c_max[ii]-c_min[ii])
        else:
            raise ValueError('Wrong Norm')
    elif param_struct.ind_norm == 2 and param_struct.shuffle == 2:
        norm_vec = param_struct.norm_vec
        row_num_sou = param_struct.N/param_struct.m
        row_index_sou = np.floor((param_struct.var-1)/row_num_sou)+1
        c_norm = np.zeros(param_struct.M, 1)
        for ii in range(1, param_struct.M):
            if norm_vec[ii] == 's':
                c_norm[ii] = (1/c[ii] - 1/c_min[ii, row_index_sou]) \
                             / (1/c_max[ii, row_index_sou]-1/c_min[ii, row_index_sou])
            elif norm_vec[ii] == 'p':
                c_norm[ii] = (c[ii]-c_min[ii, row_index_sou])\
                             / (c_max[ii, row_index_sou]-c_min[ii, row_index_sou])
            else:
                raise ValueError('Wrong Norm')
    else:
        raise ValueError('Wrong Norm')
    return c_norm


def landweber(param_struct, c_m_min, c_m_max, c_measure, norm_struct, s_mat, a_lw, iter_i):
    c = cap_norm(param_struct, c_m_min, c_m_max, c_measure, norm_struct)
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
param = read_matlab_struct(datapath + 'param.mat')['param']
norm = param.norm
anim = param.anim


cap_min = np.array(read_cap_file(datapath + 'c_min_ij.txt', param))
cap_max = np.array(read_cap_file(datapath + 'c_max_ij.txt', param))
c_array = np.array(read_cap_file(datapath + 'Herz.txt', param))
s = read_matlab_var(datapath + 'S_matrix_q_neu_richtig.mat')

lw_iter = 20
iterations = 100

# TODO: Woher c_phantom?
solved = landweber(param, cap_min, cap_max, c_array, norm, s, lw_iter, iterations)