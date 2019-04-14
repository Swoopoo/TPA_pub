##!/bin/python3
import numpy as np
import landweberHelper as lH
import matplotlib.pyplot as plt

# Based on the paper 'Neural network based multi-criterion
# optimization image reconstruction
# technique for imaging two- and
# three-phase flow systems using electrical
# capacitance tomography'
# -----------------------------------------------------
# from W. Warsito und L-S Fan
# -----------------------------------------------------
# -----------------------------------------------------
#
# STAND: 16.10.2018
#
# DIE INITIALISIEREN v(0) AUF 1/n mit n = Pixelzahl eines Bildes!
#
#
#
# Initialize the model
class InitModel:

    def __init__(self, s_matrix, imagesize, deltat=0.01, tau=1, alpha_0=7, beta=2, eta=1, xi=1.9, zeta=5,
                 smoothing_weights=(1, -1/8, -1/8), w=(1/3, 1/3, 1/3)):
        self.t = 0	 # Time
        self.deltat = deltat  # Algorithm Steplength
        self.tau = tau  # = R_0 C_0 # Time Constant of the capacitors - set to 1.0 to measure in units \tau
        self.alpha_0 = alpha_0  # penalty Parameter
        self.beta = beta  # Steepness gain factor
        self.xi = xi  # y-Axis intercept of the linear activation function
        self.zeta = zeta
        self.eta = eta
        self.gam = np.zeros(3)  # Initialize Gamma_1,2,3 to zero
        self.u = np.reshape(np.zeros(imagesize[0] * imagesize[1]), [imagesize[0] * imagesize[1], 1])
        self.u_deltat = np.reshape(np.zeros(imagesize[0] * imagesize[1]), [imagesize[0] * imagesize[1], 1])
        self.v = self.activation(self.u)  # Initialize Node Outputs
        self.G_width = imagesize[1]
        self.G_height = imagesize[0]
        self.S = s_matrix
        self.smoothing_weights = smoothing_weights  # Smoothing Weights for smoothing the Image vector
        self.w = np.array(w)  # Initial Values vor omega_1,2,3

    # Differential von u nach S. 2205 (31)
    # Ist self.gam hier wirklich das gleiche wie in den anderen Funktionen? Im
    # Text steht auf S. 2204 das gamma = gamma / C_0 substituiert wird.

    def getNeighbours(self, i):
        """Find neighbours for the smoothie calculation (lule)"""
        E = []
        V = []
        i_row = i // self.G_width
        i_col = i - self.G_width * i_row # faster than modulo
        if i_row != 0:
            E.append(i - self.G_width)
            if i_col != 0:
                V.append(i - self.G_width - 1)
            if i_col != self.G_width - 1:
                V.append(i - self.G_width + 1)
        if i_row != self.G_height - 1:
            E.append(i + self.G_width)
            if i_col != 0:
                V.append(i + self.G_width - 1)
            if i_col != self.G_width - 1:
                V.append(i + self.G_width + 1)
        if i_col != 0:
            E.append(i - 1)
        if i_col != self.G_width - 1:
            E.append(i + 1)
        return (E, V)

    def penaltyAlpha(self):
        """Penalty Parameter alpha according to equation 29 (p.2204)"""
        return self.alpha_0 + self.zeta * np.exp(-1 * self.eta * self.t)

    def calcZ(self):
        """Calculate z according to equation 28 (p.2204)"""
        z = np.dot(self.S, self.G) - self.C
        return z

    def calcDeltaZ(self):
        """Calculate delta(z) according to equation 27 (p.2204)"""
        z = self.calcZ()
        deltaz = self.penaltyAlpha() * z
        deltaz[z <= 0] = 0
        return deltaz

    def smoothie(self, G):
        """Calculate matrixproduct X*G"""
        XG = self.smoothing_weights[0] * G.copy()
        for i in range(XG.size):
            E, V = self.getNeighbours(i) # returns 2 lists of indices
            XG[i] +=   self.smoothing_weights[1] * np.sum(G[E]) \
                     + self.smoothing_weights[2] * np.sum(G[V])
        return XG

    def calcGamma(self):
        """Calculate gamma values for the algorithm according to equations on page 2207"""
        print('Calculate Gammas')
        # self.gam[0] = 1 /np.dot(self.G_1darray, np.log(self.G_1darray))
        # self.gam[0] = 1 /np.dot(np.ndarray.flatten(self.G), np.ndarray.flatten(np.log(self.G)))
        self.gam[0] = 1 /self.flat_dot(self.G, np.log(self.G))
        # self.gam[0] = 1 /np.dot(self.G.flatten(), np.log(self.G).flatten())
        self.gam[1] = 2 / (np.linalg.norm(np.dot(self.S, self.G) - self.C)**2)
        # self.gam[2] = 1 / ((0.5 * np.dot(self.G.T, self.smoothie(self.G)) + 0.5 * np.dot(self.G.T, self.G)))
        self.gam[2] = 2 / (self.flat_dot(self.G, self.smoothie(self.G)) + self.flat_dot(self.G, self.G))

    def func1(self, G):
        """Objectfunction f1 according to equation 14 (p. 2203)"""
        # G_1darray = np.reshape(G, G.shape[0])
        # f1 = self.gam[0] * np.dot(G_1darray,np.log(G_1darray))
        # f1 = self.gam[0] * np.dot(np.ndarray.flatten(G),np.log(np.ndarray.flatten(G)))
        f1 = self.gam[0] * self.flat_dot(G, np.log(G))
        return f1

    def func2(self, G):
        """Objectfunction f2 according to equation 15 (p. 2203)"""
        f2 = 0.5 * self.gam[1] * (np.linalg.norm(np.dot(self.S, G) - self.C))**2
        return f2

    def func3 (self, G):
        """Objectfunction f3 according to equation 16 (p. 2203)"""
        # f3 = 0.5 * self.gam[2] * (np.dot(G.T, self.smoothie(G)) + np.dot(G.T, G))
        f3 = 0.5 * self.gam[2] * (self.flat_dot(G, self.smoothie(G)) + self.flat_dot(G, G))
        return f3

    def deltaWeights(self, n):
        """Calculate weights 1-3 according to p.2207"""
        if n == 1:
            return self.func1(self.deltaG()) - self.func1(self.G)
        elif n == 2:
            return self.func2(self.deltaG()) - self.func2(self.G)
        elif n == 3:
            return self.func3(self.deltaG()) - self.func3(self.G)

    def updateWeights(self):
        """Update weights according to the updatestep given on p.2207"""
        print('Updating Weights')
        weightsum = 0
        for i in range(3):
            weightsum += (self.deltaWeights(1) / self.deltaWeights(i+1))
        for i in range(3):
            self.w[i] = (self.deltaWeights(1) / self.deltaWeights(i+1))/weightsum

    def timestep(self):
        """Perform a step in the internal algorithm time"""
        # deltaG is dependent on the u of the current time step so must be done
        # before u is updated
        #self.G = self.deltaG()
        #self.u = self.deltaU()
        #self.t += self.deltat
        # Perhaps a more readable or more transparent implementation, the usage
        # of deltaG and deltaU in the above is too confusing.
        self.u += self.derivu() * self.deltat
        self.G  = self.activation(self.u)
        self.t += self.deltat
        print('Timestep')

    def deltaG(self):
        """Calculate change in image vector"""
        return self.activation(self.deltaU())

    def deltaU(self):
        """Calculate G and u at the next timestep according to equation"""
        return self.u + self.derivu() * self.deltat

    def derivu(self):
        """Calculate the derivative of u according to equation 31 (p.2205)"""
        u_deriv = -(self.u/self.tau) - (
                self.w[0] * self.gam[0] * (1 + np.log(self.G))
                + self.w[1] * self.gam[1] * np.dot(self.S.T, self.calcZ())
                + self.w[2] * self.gam[2] * (self.smoothie(self.G) + self.G)
                + np.dot(self.S.T, self.calcDeltaZ())
        )
        return u_deriv

    def updateImage(self):
        """Calculate image vector according to equation 36"""
        self.G = self.deltaG()
        print('Update Image')
        plt.imshow(np.reshape(self.G, (91, 91)))
        plt.savefig('/home/andre/Documents/Git/TPA_pub/Andre/testpics/'+str(self.t)+'.png')
        plt.close()

    def initializeG(self, C_m_min, C_m_max, param):
        """Initialize first Imagevector G"""
        # return lH.landweber(param, C_m_min, C_m_max, self.C, param.norm, self.S, 5, 1000)
        #return np.dot(self.S.T, self.C)
        return self.activation(self.u)

    def activation(self, u):
        """Activation function"""
        v = self.beta * u + self.xi
        v[u <= -self.xi/self.beta] = 0
        v[u >= 1 - self.xi/self.beta] = 1
        return v

    def updatingStep(self):
        """Run an updating step according to p.2207"""
        # TODO: This looks very wrong. Updating should be done in timestep,
        # that's what it's for. Later both updatingStep and timestep are used in
        # calc_G in each iteration. updatingStep calls updateImage which does a
        # new calculation of G. Maybe I'm too tired to understand this...
        self.calcGamma()
        self.updateWeights()
        self.updateImage()

    def calcError(self):
        """Calculate the mean square error of the image vector"""
        return np.linalg.norm(self.deltaG() - self.G)**2

    def calc_G(self, C_phantom, C_min, C_max, param, init=0):
        """Calculate the image vector G from the given capacity values C"""
        self.C = C_phantom
        if init == 1:
            self.G = self.initializeG(C_min, C_max, param)
        # self.G_1darray = np.reshape(self.G, (len(self.G)))
        self.updatingStep()
        Error = self.calcError()
        print(Error)
        if len(np.where(Error > 1e-4)[0]) == 0:
            print('Fertig nach '+str(self.t)+' Iterationen')
            self.plotG()
        else:
            self.timestep()
            print('Iteration: ' +str(self.t*100))
            self.calc_G(C_phantom, C_min, C_max, param, init=0)

    def plotG(self):
        plt.imshow(np.reshape(self.deltaG(), (91, 91)))
        plt.colorbar()
        plt.show()
    
    def flat_dot(A,B):
        return np.dot(np.ndarray.flatten(A),np.ndarray.flatten(B))


# if __name__ == '__main__':
pixels = (91, 91)

phantomdir = 'mulkreis'
phantompath = '/home/andre/Documents/Studium/Teamprojektarbeit/Matlab Toolbox/Matlab/' + phantomdir + '/c_phantom3.txt'
datapath = '/home/andre/Documents/Studium/Teamprojektarbeit/Datenundshit/Daten_TPA_23012019/TPA_Daten_fixed.mat'
minpath = '/home/andre/Documents/Studium/Teamprojektarbeit/Matlab Toolbox/Matlab/' + phantomdir + '/c_min_ij.txt'
maxpath = '/home/andre/Documents/Studium/Teamprojektarbeit/Matlab Toolbox/Matlab/' + phantomdir + '/c_max_ij.txt'

import_struct = lH.read_matlab_struct(datapath)
param = import_struct['param']
s_mat = import_struct['solu'].S_r

import_phantom = lH.read_cap_file(phantompath, param)
import_phantom_min = lH.read_cap_file(minpath, param)
import_phantom_max = lH.read_cap_file(maxpath, param)


Model = InitModel(s_mat, pixels)
ImageSolution = Model.calc_G(import_phantom, import_phantom_min, import_phantom_max, param, init=1)
# NNMOIRT mieft
# print(ImageSolution)
# G = np.dot(S.T, C)
