import scipy.io as sio
import pandas as pd


def read_matlab_struct(path):
    # Speichert die Struct als Python Object
    return sio.loadmat(path, struct_as_record=False, squeeze_me=True)


def read_matlab_var(path):
    var = sio.loadmat(path)
    return var.get(list(var.keys())[-1])


def read_cap_file(path):
    return pd.read_csv(path, header=None)
