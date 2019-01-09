import scipy.io as sio
import pandas as pd
import numpy as np


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


def read_matlab_struct(path):
    """Import matlab struct as python object"""
    return sio.loadmat(path, struct_as_record=False, squeeze_me=True)


def read_matlab_var(path):
    """Import matlab variables as numpy arrays"""
    var = sio.loadmat(path)
    return var.get(list(var.keys())[-1])


def read_cap_file(path):
    """Import capacity data from .txt file"""
    data_array = np.array(pd.read_csv(path, header=None, decimal='.', delim_whitespace=True))
    indx = np.tril_indices(data_array.shape[0], -1)
    c_array = data_array[indx]
    return np.reshape(c_array, [len(c_array), 1])
