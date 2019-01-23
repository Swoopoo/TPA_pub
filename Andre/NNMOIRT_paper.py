##!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import landweberHelper as lH

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

    def __init__(self, C, S, ImageSize, deltat = 0.01, tau = 1, alpha_0 = 7, beta = 2, eta = 1, xi = 0.55555, zeta = 5,
                 smoothing_weights = (1, -1/8, -1/8), w = (1/3, 1/3, 1/3)):
        self.t = 0	 # Time
        self.deltat = deltat # Algorithm Steplength
        self.tau = tau# = R_0 C_0 # Time Constant of the capacitors - set to 1.0 to measure in units \tau
        self.alpha_0 = alpha_0 # penalty Parameter
        self.beta = beta # Steepness gain factor - vertical slope and the horizontal spread of the sigmoid-shape function
        self.xi = xi # y-Axis intercept of the linear activation function
        self.zeta = zeta
        self.eta = eta
        self.gam = np.zeros(3) # Initialize Gamma_1,2,3 to zero
        self.u = np.reshape(np.zeros(ImageSize[0] * ImageSize[1]), [ImageSize[0] * ImageSize[1], 1]) # Init. of Network Input Vector
        self.u_deltat = np.reshape(np.zeros(ImageSize[0] * ImageSize[1]), [ImageSize[0] * ImageSize[1], 1])
        self.v = self.activation(self.u) # Initialize Node Outputs
        self.G_width = ImageSize[1]
        self.G_height = ImageSize[0]
        self.C = C
        self.S = S
        self.smoothing_weights = smoothing_weights# Smoothing Weights for smoothing the Image vector
        self.w = np.array(w) # Initial Values vor omega_1,2,3

    # Differential von u nach S. 2205 (31)
    # Ist self.gam hier wirklich das gleiche wie in den anderen Funktionen? Im
    # Text steht auf S. 2204 das gamma = gamma / C_0 substituiert wird.


    def get_neighbours(self, i):
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
        deltaz = self.penaltyAlpha() * self.calcZ()
        deltaz[self.calcZ() <= 0] = 0
        return deltaz


    def smoothie(self, G):
        """Calculate matrixprodukt X*G"""
        XG = self.smoothing_weights[0] * G.copy()
        for i in range(XG.size):
            E, V = self.get_neighbours(i) # returns 2 lists of indices
            XG[i] +=   self.smoothing_weights[1] * np.sum(G[E]) \
                     + self.smoothing_weights[2] * np.sum(G[V])
        return XG


    def derivu(self):
        """Calculate the derivative of u according to equation 31 (p.2205)"""
        u_deriv = -(self.u/self.tau) - (
                  self.w[0] * self.gam[0] * (1 + np.log(self.G))
                + self.w[1] * self.gam[1] * np.dot(self.S.T, self.calcZ())
                + self.w[2] * self.gam[2] * (self.smoothie(self.G) + self.G)
                + np.dot(self.S.T, self.calcDeltaZ())
                )
        return u_deriv


    def calcGamma(self):
        """Calculate gamma values for the algorithm according to equations on page 2207"""
        print('Calculate Gammas')
        gamsum = np.dot(self.G_1darray, np.log(self.G_1darray))
        self.gam[0] = 1 / gamsum
        self.gam[1] = 1 / ((0.5 * np.linalg.norm(np.dot(self.S, self.G) - self.C))**2)

        self.gam[2] = 1 / ((0.5 * np.dot(self.G.T, self.smoothie(self.G)) + 0.5 * np.dot(self.G.T, self.G)))


    def deltaU(self):
        """Calculate G and u at the next timestep according to equation"""
        return self.u + self.derivu() * self.deltat


    def deltaG(self):
        """Calculate change in image vector"""
        return self.activation(self.deltaU())


    def func1(self, G):
        """Objectfunction f1 according to equation 14 (p. 2203)"""
        G_1darray = np.reshape(G, G.shape[0])
        f1 = self.gam[0] * np.dot(G_1darray,np.log(G_1darray))
        return f1


    def func2(self, G):
        """Objectfunction f2 according to equation 15 (p. 2203)"""
        f2 = 0.5 * self.gam[1] * np.linalg.norm(np.dot(self.S, G) - self.C)
        return f2


    def func3 (self, G):
        """Objectfunction f3 according to equation 16 (p. 2203)"""
        f3 = 0.5 * self.gam[2] * (np.dot(G.T, self.smoothie(G)) + np.dot(G.T, G))
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
        for i in range(3):
            weightsum = 0
            for k in range(3):
                weightsum += (self.deltaWeights(1) / self.deltaWeights(k+1))
            self.w[i] = (self.deltaWeights(1) / self.deltaWeights(i+1))/weightsum


    def timestep(self):
        """Perform a step in the internal algorithm time"""
        self.t += self.deltat
        self.u = self.deltaU()
        self.G = self.deltaG()
        print('Timestep')


    def updateImage(self):
        """Calculate image vector according to equation 36"""
        self.G = self.deltaG()
        print('Update Image')


    def initializeG(self):
        """Initialize first Imagevector G"""
        return self.activation(self.u)


    def activation(self, u):
        """Activation function"""
        beta = 1
        xi = 1
        v = beta * u + xi
        v[u <= -xi/beta] = 0
        v[u >= 1 - xi/beta] = 1
        return v


    def updatingStep(self):
        """Run an updating step according to p.2207"""
        self.calcGamma()
        self.updateWeights()
        self.updateImage()


    def calcError(self):
        """Calculate the mean square error of the image vector"""
        return np.linalg.norm(self.deltaG() - self.G)**2


    def calc_G(self, C):
        """Calculate the image vector G from the given capacity values C"""
        self.C = C
        self.G = self.initializeG()
        self.G_1darray = np.reshape(self.G, (len(self.G)))
        self.updatingStep()
        Error = self.calcError()
        if len(np.where(Error > 1e-4)[0]) == 0:
            return self.deltaG()
        else:
            self.timestep()
            self.calc_G(C)



#if __name__ == '__main__':
ImageSize = (91, 91)

phantomdir = 'mulkreis'
phantompath = '/home/andre/Documents/Studium/Teamprojektarbeit/Matlab Toolbox/Matlab/' + phantomdir + '/c_phantom3.txt'
datapath = '/home/andre/Documents/Studium/Teamprojektarbeit/Datenundshit/Daten_TPA_23012019/TPA_Daten_fixed.mat'
import_struct = lH.read_matlab_struct(datapath)
param = import_struct['param']
S = import_struct['solu'].S_r
C = lH.read_cap_file(phantompath, param)


Model = InitModel(C, S, ImageSize)
ImageSolution = Model.calc_G(C)
#print(ImageSolution)
#G = np.dot(S.T, C)
