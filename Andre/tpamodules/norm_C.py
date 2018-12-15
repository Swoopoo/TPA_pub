import numpy as np

# functions in this file override the values in input C

def norm_C(C, C_max, C_min):
    if C.size != C_max.size or C.size != C_min.size:
        raise ValueError('Sizes of C, C_max and C_min do not match')
    for i in range(C.size): C[i] = ( C[i] - C_min[i] ) / ( C_max[i] - C_min[i] )
    return C

def norm_C_simple(C, C_max, C_min):
    if type(C_max) != float and type(C_max) != np.float64:
        C_max = np.max(C_max)
    if type(C_min) != float and type(C_min) != np.float64:
        C_min = np.min(C_min)
    C = C - C_min
    C = C / ( C_max - C_min )
    return C
