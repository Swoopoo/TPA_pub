import scipy.io as sio
import pandas as pd
import numpy as np


def read_matlab_struct(path):
    # Speichert die Struct als Python Object
    return sio.loadmat(path, struct_as_record=False, squeeze_me=True)


def read_matlab_var(path):
    var = sio.loadmat(path)
    return var.get(list(var.keys())[-1])


def read_cap_file(path, param_struct):
    data_array = np.array(pd.read_csv(path, header=None, decimal='.', delim_whitespace=True))
    c_array = c_array_conversion(data_array, param_struct)
    return c_array


def c_array_conversion(c_array, param_struct):
    if param_struct.shuffle == 1:
        c = np.zeros((param_struct.M, 1))
        pp = 1
        for ii in range(0, (param_struct.m-1)):
            for jj in range(ii, param_struct.m):
                c[pp] = c_array[ii, jj]
                pp += 1
    elif param_struct.shuffle == 2:
        i_end_in = param_struct.m
        i_end_out = np.int(np.floor((param_struct.m-1)/2))
        i_c = 1
        c = np.zeros((param_struct.M, 1))
        for i_out in range(0,i_end_out):
            for i_in in range(0, i_end_in):
                sec_ind = i_in + i_out
                if sec_ind > param_struct.m:
                    sec_ind = np.mod(sec_ind, param_struct.m)
                c[i_c] = c_array[i_in, sec_ind]
                i_c += 1
        if np.mod(param_struct.m, 2) == 0:
            for i_in in range(0, i_end_in/2):
                sec_ind = i_in+param_struct.m/2
                c[i_c] = c_array[i_in, sec_ind]
                i_c += 1
    else:
        raise ValueError('Wrong number in param.shuffle, must be 1 or 2')
    return c
