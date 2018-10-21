from scipy.io import loadmat
import numpy as np

def importMatFile(pfad):
    # f = h5py.File(pfad,'r') Für Matlab Versionen 7.3+ (ab Dezember 2015)
    S = loadmat(pfad)
    mat_keys = list(S)
    return S[mat_keys[-1]][0,0][0]