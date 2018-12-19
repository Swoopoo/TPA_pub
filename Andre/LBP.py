import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

def capNorm(param, C_min, C_max,C,norm):
    if param.ind_norm == 1:
        if norm == 's':
            C_norm = np.zeros(param.M, 1)
            for ii in range(1, param.M):
                C_norm[ii] = (1/C[ii]-1/C_min[ii])/(1/C_max[ii]-1/C_min[ii])
        elif norm == 'p':
            C_norm = np.zeros(param.M, 1)
            for ii in range(1, param.M):
                C_norm[ii] = (C[ii]-C_min[ii])/(C_max[ii]-C_min[ii])
        else:
            raise ValueError('Wrong Norm')
    elif param.ind_norm == 2 and param.shuffle == 2:
        norm_vec =param.norm_vec
        row_num_sou = param.N/param.m
        row_index_sou = np.floor((param.var-1)/row_num_sou)+1
        C_norm = np.zeros(param.M, 1)
        for ii in range(1,param.M):
            if norm_vec[ii] == 's':
                C_norm[ii] = (1/C[ii] - 1/C_min[ii, row_index_sou]) \
                             /(1/C_max[ii,row_index_sou]-1/C_min[ii,row_index_sou])
            elif norm_vec[ii] == 'p':
                C_norm[ii] = (C[ii]-C_min[ii,row_index_sou])\
                             /(C_max[ii,row_index_sou]-C_min[ii,row_index_sou])
            else:
                raise ValueError('Wrong Norm')
    else:
        raise ValueError('Wrong Norm')
    return C_norm



def PLW(param, cmats, S, a_lw, iter_i):

    C_phantom = cmats.C_phantom
    C = capNorm(param, cmats.C_m_min, cmats.C_m_max, C_phantom, norm)

    g = np.dot(S.T, C)
    N = g.size[0]

    Z = np.zeros(1, param.nlist.size[0])

    if anim == 1:
        for ii in range(1, iter_i):
            g = g + a_lw * np.dot(S.T, (C-np.dot(S,g)))
            g[g<0] = 0
            g[g>=1] = 1
    elif anim == 0:
        for ii in range(1, iter_i):
            g = g + a_lw * np.dot(S.T, (C-np.dot(S,g)))
            g[g<0] = 0
            g[g>=1] = 1
    return g
param = sio.loadmat('param.mat', struct_as_record = False, squeeze_me = True) # Speichert die Struct als Python Object
cmats = sio.loadmat('cmats.mat', struct_as_record = False, squeeze_me = True) # Speichert die Struct als Python Object
norm = param.norm
anim = param.anim
a_lw = 20
iter_i = 100
g = PLW(param, cmats, S, a_lw, iter_i)
