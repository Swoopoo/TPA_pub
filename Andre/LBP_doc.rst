Landweber Algorithmus in Python
===============================
Helper functions
----------------

**Import Matlab variables into python**

.. code-block:: python

    def read_matlab_var(path):
        var = sio.loadmat(path)
        return var.get(list(var.keys())[-1])

**Import Matlab struct into python**
c_phantom
.. code-block:: python

    def read_matlab_struct(path):
        return sio.loadmat(path, struct_as_record=False, squeeze_me=True)

**Import ANSYS output .txt file into python**

.. code-block:: python

    def read_cap_file(path):
        return pd.read_csv(path, header=None)

Landweber Algorithm
-------------------

**Normalize capacity values**

.. code-block:: python

    def cap_norm(param_struct, c_min, c_max, c, norm_struct):
        if param_struct.ind_norm == 1:
            if norm_struct == 's':
                c_norm = np.zeros(param_struct.M, 1)
                for ii in range(1, param_struct.M):
                    c_norm[ii] = (1/c[ii]-1/c_min[ii])/(1/c_max[ii]-1/c_min[ii])
            elif norm_struct == 'p':
                c_norm = np.zeros(param_struct.M, 1)
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

param_struct
    work_path
        Arbeitspfad (char)
    param.nlist_nam
        Name der Knotendatei
    param.elist_name
        Name der Elementdatei
    ansys_cmats_dir
        Ordnername der Ansys-Kapazitätsmatrizen (char)
    er_max
        Maximale Permittivität (double)
    er_min
        Minimale Permittivität (double)
    m
        Anzahl der Elektroden (int)
    N
        Anzahl der Pixel (int)
    M
        Anzahl der unab. Elektrodenpaarungen (int)
    norm
        Art der Normierung. 's' oder 'p' (char)
    anim
        Animationsoption. 1: Mit 0: Ohne (bin)
    ref
        Referenzkap. der Messkarte PicoCap (double)
    elist
        Koinzidenzliste der Pixel (N x 3 - int-Array)
    nlist
        Knotenkoordinaten (? x 3 - double-Array)
c_min
    Ansys-Kapazitätsmatrix bei er_min (m x m)
c_max
    Ansys-Kapazitätsmatrix bei er_max (m x m)
c
    Kapazitätsvektor mit Phantom (m x 1)
norm_struct
    Art der Normierung
        'p'
            Normierung auf Maximum und Minimum
        's'
            Quotientennormierung auf Maximum und Minimum

**Run landweber algorithm**

.. code-block:: python

    def landweber(param_struct, c_m_min, c_m_max, c_phantom, norm_struct, s_mat, a_lw, iter_i):
        c = cap_norm(param_struct, c_m_min, c_m_max, c_phantom, norm_struct)
        g = np.dot(s_mat.T, c)
        # N = g.size[0]
        # Z = np.zeros(1, param_struct.nlist.size[0])
        if anim == 1:
            for ii in range(1, iter_i):
                g = g + a_lw * np.dot(s_mat.T, (c-np.dot(s_mat, g)))
                g[g < 0] = 0
                g[g >= 1] = 1
        elif anim == 0:
            for ii in range(1, iter_i):
                g = g + a_lw * np.dot(s_mat.T, (c-np.dot(s_mat, g)))
                g[g < 0] = 0
                g[g >= 1] = 1
        return g

s_mat
    Generierte S Matrix
a_lw
    Schrittweite des Algorithmus
iter_i
    Anzahl der Iterationen