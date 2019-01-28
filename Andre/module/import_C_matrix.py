import pandas as pd
import numpy as np

def importCMatrix(pfad):
    C = np.array(pd.read_csv(pfad, header = None, delimiter = r"\s+"))
    indx = np.tril_indices(C.shape[0], -1)
    C_array = C[indx]
    return np.reshape(C_array, [len(C_array), 1])
