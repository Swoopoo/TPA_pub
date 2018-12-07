import numpy as np

class PLW:
    # init function {{{
    def __init__ ( self, S, a_lw
            ,animated = False
            # ,elist
            # ,nlist
            # ,norm
            # ,C_m_max
            # ,C_m_min
            ):
        self.S = S
        self.a_lw = a_lw
        self.animated = animated
        # self.elist = elist
        # self.nlist = nlist
        # self.norm = norm
        # self.C_m_max = C_m_max
        # self.C_m_min = C_m_min
        return
    def plw ( self, C_phantom, iterations, a_lw = False, animated = False):
        # assuming C_phantom has been normed
        if a_lw != False: self.a_lw = a_lw
        if animated != False: self.animated = animated
        g = np.dot(self.S.T, C_phantom)
        N = g.shape[0]
        # Z = np.zeros((1,self.nlist.shape[0]))
        if self.animated:
            for i in range(iterations):
                g += self.a_lw * np.dot(self.S.T, C_phantom - np.dot(self.S, g))
                g[np.where(g < 0)] = 0
                g[np.where(g > 1)] = 1
                self.update_animation(Z,g)
        else:
            for i in range(iterations):
                g += self.a_lw * np.dot(self.S.T, C_phantom - np.dot(self.S, g))
                g[np.where(g < 0)] = 0
                g[np.where(g > 1)] = 1
        return g
# def PLW(param, cmats, S, a_lw, iterations):
    # Input as in Matlab script for now:
    #    S = Sensitvitätsmatrix
    #    a_lw = Schrittweite
    #    iter = Anzahl der Iterationen
    #    param = 
    #             work_path: Arbeitspfad (char)
    #       param.nlist_nam: Name der Knotendatei
    #      param.elist_name: Name der Elementdatei
    #       ansys_cmats_dir: Ordnername der Ansys-Kapazitätsmatrizen (char)
    #                er_max: Maximale Permittivität (double)
    #                er_min: Minimale Permittivität (double)
    #                     m: Anzahl der Elektroden (int)
    #                     N: Anzahl der Pixel (int)
    #                     M: Anzahl der unab. Elektrodenpaarungen (int)
    #                  norm: Art der Normierung. 's' oder 'p' (char)
    #                  anim: Animationsoption. 1: Mit 0: Ohne (bin)
    #                   ref: Referenzkap. der Messkarte PicoCap (double)
    #                 elist: Koinzidenzliste der Pixel (N x 3 - int-Array)
    #                 nlist: Knotenkoordinaten (? x 3 - double-Array)
    #    cmats = 
    #                C_max: Ansys-Kapazitätsmatrix bei er_max (m x m)
    #                C_min: Ansys-Kapazitätsmatrix bei er_min (m x m)
    #                  C_n: Cell mit N Ansys-Kapazitätsmatrizen (1 x N)
    #              C_m_max: Kapazitätsvektor bei er_max (M x 1)
    #              C_m_min: Kapazitätsvektor bei er_min (M x 1)
    #          C_m_phantom: Kapazitätsvektor mit Phantom (M x 1)
