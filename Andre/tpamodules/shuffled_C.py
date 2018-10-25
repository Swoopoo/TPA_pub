import pandas as pd
import numpy as np

def read_and_shuffle_C_matrix(path, OneDimension=False):
    C = np.array(pd.read_csv(path, header = None, delimiter = r"\s+"))
    if C.shape[0] != C.shape[1]:
        raise ValueError('Matrix contained in "%s" is no square matrix'%(path))
    count = C.shape[0]
    size = count * ( count - 1 ) // 2
    C_out = np.zeros(size)

    # Implementation mindlessly copied from cmat_ansys_import.m by A. Bogner and
    # M. Mösch
    i_end_out = ( count - 1 ) // 2
    i_c = 1
    for i_out in range(1, i_end_out+1):
        for i_in in range(1, count+1):
            sec_ind = i_in + i_out
            if sec_ind > count: sec_ind = sec_ind % count
            C_out[i_c-1] = C[i_in-1, sec_ind-1]
            i_c += 1
    if count%2 == 0:
        for i_in in range(1,count//2+1):
            sec_ind = i_in + count//2
            C_out[i_c-1] = C[i_in-1, sec_ind-1]
            i_c += 1
    if OneDimension:
        return C_out # return the same shape as input data had
    else:
        return np.reshape(C_out, (size, 1))
