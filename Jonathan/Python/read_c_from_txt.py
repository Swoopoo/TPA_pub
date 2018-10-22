import numpy as np

def readCfromTXT(input_file):
    full = np.loadtxt(input_file)
    sliced = [ full[i][j]
            for j in range(len(full)) for i in range(j+1,len(full)) ]
    return np.array(sliced)
