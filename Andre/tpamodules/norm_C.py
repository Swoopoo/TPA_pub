def norm_C(C, C_max, C_min):
    if C.size != C_max.size or C.size != C_min.size:
        raise ValueError('Sizes of C, C_max and C_min do not match')
    for i in range(C.size): C[i] = ( C[i] - C_min[i] ) / ( C_max[i] - C_min[i] )
    return C
