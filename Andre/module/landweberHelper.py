import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def landweber(param_struct, c_m_min, c_m_max, c_measure, norm_struct, s_mat, a_lw, iter_i):
    """Execute Landweber algorithm"""
    c = cap_norm(param_struct, c_m_min, c_m_max, c_measure, norm_struct)
    g = np.dot(s_mat.T, c)
    if param_struct.anim == 1:
        for ii in range(0, iter_i):
            g = g + a_lw*np.dot(s_mat.T, (c - np.dot(s_mat,g)))
            g[g < 0] = 0
            g[g >= 1] = 1
        plot_ect(g)
    elif param_struct.anim == 0:
        for ii in range(0, iter_i):
            g = g + a_lw*np.dot(s_mat.T, (c - np.dot(s_mat,g)))
            g[g < 0] = 0
            g[g >= 1] = 1
            print(ii)
    return g


def cap_norm(param_struct, c_min, c_max, c_measure, norm_struct):
    """ Normalize the capacity values"""
    c = c_measure
    if param_struct.ind_norm == 1:
        c_norm = np.zeros((param_struct.M, 1))
        if norm_struct == 's':
            for ii in range(0, param_struct.M):
                c_norm[ii] = (1/c[ii]-1/c_min[ii])/(1/c_max[ii]-1/c_min[ii])
        elif norm_struct == 'p':
            c_norm = np.divide((c_measure - c_min),(c_max - c_min))
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


def read_matlab_struct(path):
    """Import matlab struct as python object"""
    return sio.loadmat(path, struct_as_record=False, squeeze_me=True)


def read_matlab_var(path):
    """Import matlab variables as numpy arrays"""
    var = sio.loadmat(path)
    return var.get(list(var.keys())[-1])


def read_cap_file(path, param):
    """Import capacity data from .txt file"""
    data_array = np.array(pd.read_csv(path, header=None, decimal='.', delim_whitespace=True))
    C = np.zeros((param.M, 1))
    pp = 0
    for ii in range(0, param.m-1):
        for jj in range(ii+1, param.m):
            C[pp] = data_array[ii, jj]
            pp += 1
    return C


def plot_ect(S_mat):
    fig = plt.figure()
    pl1 = plt.imshow(np.reshape(S_mat, (91, 91)))
    plt.show()