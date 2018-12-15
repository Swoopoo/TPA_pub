##!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from tpamodules.import_S_matrix import importMatFile
from tpamodules.import_C_matrix import importCMatrix

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
# TODO: Rest der Funktionen (23+) durchsehen ob passen
#
# Notizen: Die Normierung bei u' kommt daher: Neural Network Approach ... (eq. 13) <- TODO: DAS PAPER UNBEDINGT LESEN
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
        self.zeta = zeta # TODO: ??
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
    # TODO: Gamma is normalized on the Reference Capacity, but C_0 = 1 so we'll leave it at that for now
    # TODO: Problem: Log(G) geht nicht, da G auf zeros initialisiert wird!!!!


    # Find neighbours for the smoothie calculation (lule)
    def get_neighbours(self, i):
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


    # Penalty Parameter alpha according to equation 29 (p.2204)
    def penaltyAlpha(self):
        return self.alpha_0 + self.zeta * np.exp(-1 * self.eta * self.t)


    # Calculate z according to equation 28 (p.2204)
    def calcZ(self):
        z = np.dot(self.S, self.G) - self.C
        return z


    # Calculate delta(z) according to equation 27 (p.2204)
    def calcDeltaZ(self):
        deltaz = self.penaltyAlpha() * self.calcZ()
        deltaz[self.calcZ() <= 0] = 0
        return deltaz


    # Calculate matrixprodukt X*G
    def smoothie(self, G):
        XG = self.smoothing_weights[0] * G.copy()
        for i in range(XG.size):
            E, V = self.get_neighbours(i) # returns 2 lists of indices
            XG[i] +=   self.smoothing_weights[1] * np.sum(G[E]) \
                     + self.smoothing_weights[2] * np.sum(G[V])
        return XG


     # Calculate the derivative of u according to equation 31 (p.2205)
    def derivu(self):
        u_deriv = -(self.u/self.tau) - (
                  self.w[0] * self.gam[0] * (1 + np.log(self.G))
                + self.w[1] * self.gam[1] * np.dot(self.S.T, self.calcZ())
                + self.w[2] * self.gam[2] * (self.smoothie(self.G) + self.G)
                + np.dot(self.S.T, self.calcDeltaZ())
                )
        return u_deriv


    # Calculate gamma values for the algorithm according to equations on page 2207
    def calcGamma(self):
        print('Calculate Gammas')
        gamsum = np.dot(self.G_1darray, np.log(self.G_1darray))
        self.gam[0] = 1 / gamsum
        # TODO: HIER IST GLAUB ICH EIN SCHREIBFEHLER IM PAPER, ES MueSSTE MINUS C HEIssEN DA DIE DIMENSIONEN JA GARNICHT
        # TODO: STIMMEN KoeNNEN!!!
        self.gam[1] = 1 / ((0.5 * np.linalg.norm(np.dot(self.S, self.G) - self.C))**2)

        self.gam[2] = 1 / ((0.5 * np.dot(self.G.T, self.smoothie(self.G)) + 0.5 * np.dot(self.G.T, self.G)))


    # Calculate G and u at the next timestep according to equation
    def deltaU(self):
        return self.u + self.derivu() * self.deltat


    def deltaG(self):
        return self.activation(self.deltaU())


    # Objectfunction f1 according to equation 14 (p. 2203)
    def func1(self, G):
        G_1darray = np.reshape(G, G.shape[0])
        f1 = self.gam[0] * np.dot(G_1darray,np.log(G_1darray))
        return f1


    # Objectfunction f2 according to equation 15 (p. 2203)
    def func2(self, G):
        f2 = 0.5 * self.gam[1] * np.linalg.norm(np.dot(self.S, G) - self.C)
        return f2


    # Objectfunction f3 according to equation 16 (p. 2203)
    def func3 (self, G):
        f3 = 0.5 * self.gam[2] * (np.dot(G.T, self.smoothie(G)) + np.dot(G.T, G))
        return f3


    # Calculate weights 1-3 according to p.2207
    def deltaWeights(self, n):
        if n == 1:
            return self.func1(self.deltaG()) - self.func1(self.G)
        elif n == 2:
            return self.func2(self.deltaG()) - self.func2(self.G)
        elif n == 3:
            return self.func3(self.deltaG()) - self.func3(self.G)


    # Update weights according to the updatestep given on p.2207
    def updateWeights(self):
        print('Updating Weights')
        for i in range(3):
            weightsum = 0
            for k in range(3):
                weightsum += (self.deltaWeights(1) / self.deltaWeights(k+1))
            self.w[i] = (self.deltaWeights(1) / self.deltaWeights(i+1))/weightsum


    # Perform a step in the internal algorithm time
    def timestep(self):
        self.t += self.deltat
        self.u = self.deltaU()
        self.G = self.deltaG()
        print('Timestep')


    # Calculate image vector according to equation 36
    def updateImage(self):
        self.G = self.deltaG()
        print('Update Image')


    # Initialize first Imagevector G
    def initializeG(self):
        return self.activation(self.u)


    def activation(self, u):
       v = self.beta * u + self.xi
       v[u <= -self.xi/self.beta] = 0
       v[u >= 1 - self.xi/self.beta] = 1
       return v


    # Run an updating step according to p.2207
    def updatingStep(self):
        self.calcGamma()
        self.updateWeights()
        self.updateImage()


     # Calculate the mean square error of the image vector
    def calcError(self):
        return np.linalg.norm(self.deltaG() - self.G)**2


    # Calculate the image vector G from the given capacity values C
    def calc_G(self, C):
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
SMatrixPfad = '../Daten/S_matrix_quadratisch.mat'
#SMatrixPfad = '../Daten/Smatrix_16elek.mat'
S = importMatFile(SMatrixPfad)

CMatrixPfad = '../Daten/3_Kreise.txt'
C = importCMatrix(CMatrixPfad)

#Model = InitModel(C, S, ImageSize)
#ImageSolution = Model.calc_G(C)
#print(ImageSolution)
G = np.dot(S.T, C)
