import numpy as np

def norm_C(C, C_max, C_min, norm_type='p'):
    if C.size != C_max.size or C.size != C_min.size:
        raise ValueError('Sizes of C, C_max and C_min do not match')
    if norm_type == 'p':
        return [
                ( C[i] - C_min[i] ) / ( C_max[i] - C_min[i] )
                for i in range(C.size)
               ]
    elif norm_type == 's':
        return [
                ( 1/C[i] - 1/C_min[i] ) / ( 1/C_max[i] - 1/C_min[i] )
                for i in range(C.size)
               ]
    else:
        raise ValueError("Unknown norm type '%s'."%(norm_type))
    return

def norm_C_simple(C, C_max, C_min):
    if type(C_max) != float and type(C_max) != np.float64:
        C_max = np.max(C_max)
    if type(C_min) != float and type(C_min) != np.float64:
        C_min = np.min(C_min)
    return [ ( c - C_min ) / ( C_max - C_min ) for c in C ]
