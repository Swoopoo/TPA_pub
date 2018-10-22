#!/bin/python3
import numpy as np
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

# Initialize the model
class InitModel:

    def __init__(self, C, S, ImageSize, deltat = 0.01, tau = 1, alpha_0 = 7, beta = 2, eta = 1, xi = 0.1, zeta = 5,
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

    # Calculate the derivative of u according to equation 31 (p.2205)
    def derivu(self):
        u_deriv = -(self.u/self.tau) - (
                  self.w[0] * self.gam[0] * (1 + np.log(self.G))
                + self.w[1] * self.gam[1] * np.dot(self.S.T, self.calcZ())
                + self.w[2] * self.gam[2] * (self.smoothie(self.G) + self.G)
                + np.dot(self.S.T, self.calcDeltaZ())
                )
        return u_deriv


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


    # Initialize first Imagevector G
    def initializeG(self, C):
        return self.activation(self.u)
#        return np.dot(self.S.T, C)


    # Calculate G and u at the next timestep according to equation
    def delta(self):
        u_deltat = self.u + self.derivu() * self.deltat
        g_deltat = self.G + self.beta * self.derivu() * self.deltat
        return u_deltat, g_deltat


    # Perform a step in the internal algorithm time
    def timestep(self):
        self.t += self.deltat
        self.u, self.G = self.delta()
        print('Timestep')


    # Objectfunction f1 according to equation 14 (p. 2203)
    def func1(self, G):
        # Entropie Funktion des Bildes - laesst Aussage ueber die 'global smoothness' des Bildvektors treffen (14)
        # G = Bildvektor
        # len(G) = Anzahl an Neuronen
        # G(j) = Wert des j-ten Neurons
        # ga = normalisierte Konstante 0 <= ga1 <=1
        G_1darray = np.reshape(G, G.shape[0])
        f1 = self.gam[0] * np.dot(G_1darray,np.log(G_1darray))
        return f1


    # Objectfunction f2 according to equation 15 (p. 2203)
    def func2(self, G):
        # zurueckrechnen auf Kapazitaeten (15)
        # S = Sensitivitaetsmatrix
        # G = Bildvektor
        # C = Gemessene Kapazitaeten
        # ga2 = normalisierte Konstante 0 <= ga2 <= 1
        #
        # Anmerkungen:
        #  - Warum ist C in self, aber G nicht? Muesste C nicht auch extern sein?
        #    Immerhin aendert sich C wie G mit jeder Messung, waehrend S statisch
        #    ist.
        f2 = 0.5 * self.gam[1] * np.linalg.norm(np.dot(self.S, G) - self.C)
        return f2
        # Alternative Implementierung von Spratte:
        #   Datentyp abhaengig:
        #     self.S : np.array
        #     self.C : np.array
        #     G      : np.array
        # entweder (bei kleinen Arrays schneller)
        #f2 = .5 * self.gam[1] * np.sum(np.square(self.S.dot(G) - self.C))
        # oder (bei grossen Arrays schneller)
        #f2 = .5 * self.gam[1] * np.sum(np.square(self.S @ G - self.C))
        #   Matrix soll overhead haben gegenueber array
        #   (https://stackoverflow.com/a/3892639/6783373) -> array ist besser
        #     self.S : np.matrix
        #     self.C : np.matrix
        #     G      : np.matrix
        # f2 = .5 * self.gam[1] * np.sum(np.square(self.S * G - C))


    # Objectfunction f3 according to equation 16 (p. 2203)
    def func3 (self, G):  # sum of the non-uniformity and peakedness functions of an image (16) - minimiert um locale
        #  'smoothness' und kleine 'peakedness' zu gewaehrleisten
        # G = Image Vektor
        # X = N x N non-uniformity Matrix, 'smoothing' Matrix, fuer Wert[i,k] gilt (1 <= j <= N):
        #           x1  wenn    k = j
        #           x2  wenn    k E E[j]
        #           x3  wenn    k E V[j]
        #           0   wenn    sonst
        #       E[j] Indexe der Pixel, die genau EINE Kante mit dem Pixel j gemeinsam haben
        #       V[j] Indexe der Pixel, die genau EINEN Knoten mit dem Pixel j gemeinsam haben
        #       x1, x2, x3 == smoothing weights == (1, -1/8, -1/8) fuer die Standard-Smoothing Matrix
        f3 = 0.5 * self.gam[2] * (np.dot(G.T, self.smoothie(G)) + np.dot(G.T, G))
        return f3


    # Spratte: Loesung fuer Berechnung der smoothMat (S.2202):
    # Unterstellt: Rechteckiges Bild mit Breite self.G_width, G[0] in linker
    # oberer Ecke, G[1] der zweite Pixel von links in der obersten Reihe
    # (Reihenfolge eigentlich nicht wirklich wichtig, bloss G[0] in einer Ecke
    # und self.G_width die Breite in die erste Richtung -- also
    # G[self.G_width-1] der letzte Pixel in der ersten Reihe/Spalte). Die
    # Funktion skaliert allerdings relativ schlecht (O(n) = n**2)
    # Spratte: Fuer grosse Matrizen wird extrem viel Speicher benoetigt. Auf meinem
    # Laptop ist bei 200x200 Pixeln Bildgroesse schluss. Eventuell muessen wir auf
    # langsamere aber speicherfreundlichere Berechnungen umsteigen -- oder aber
    # ich schreibe doch noch ein Modul in C.
    # Die Speichermenge wechst in 4ter Potenz mit der Bildbreite, bzw.
    # quadtratisch zur Pixelzahl, wobei ein 1x1 np.array 8 byte benoetigt.
    #
    # TODO: Evtl. RAM Abfragen ob smoothMat2 benutzt werden kann
    #def smoothMat2(size,width):
        #X = np.zeros((size,size))
        #for i in range(size):
            #X[i][i] = 1 #x1
            #for j in range(i+1,size):
                #i_row = i // width # Python 3 Syntax fuer Integerdivision
                #i_col = i %  width
                #j_row = j // width # Python 3 Syntax fuer Integerdivision
                #j_col = j %  width
                #if   i_row == j_row          and abs(i_col - j_col) == 1 :
                    #X[i,j] = -1/8 #x2
                #elif abs(i_row - j_row) == 1 and i_col == j_col :
                    #X[i,j] = -1/8 #x2
                #elif abs(i_row - j_row) == 1 and abs(i_col - j_col) == 1 :
                    #X[i,j] = -1/8 #x3
                #X[j,i] = X[i,j]
        #return X
    #def neighbours(self,i,j):
        #x1 = 1
        #if i == j: return x1
        #x2 = -1/8
        #x3 = -1/8
        #i_row = i // self.G_width # Python 3 Syntax fuer Integerdivision
        #i_col = i %  self.G_width
        #j_row = j // self.G_width # Python 3 Syntax fuer Integerdivision
        #j_col = j %  self.G_width
        #if i_row == j_row          and abs(i_col - j_col) == 1 : return x2
        #if abs(i_row - j_row) == 1 and i_col == j_col          : return x2
        #if abs(i_row - j_row) == 1 and abs(i_col - j_col) == 1 : return x3
        #return 0
    # Calculate matrixprodukt X*G
    def smoothie(self, G):
        XG = self.smoothing_weights[0] * G.copy()
        for i in range(XG.size):
            E, V = self.get_neighbours(i) # returns 2 lists of indices
            XG[i] +=   self.smoothing_weights[1] * np.sum(G[E]) \
                     + self.smoothing_weights[2] * np.sum(G[V])
        return XG


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


    # Calculate gamma values for the algorithm according to equations on page 2207
    def calcGamma(self):

        gamsum = np.dot(self.G_1darray, np.log(self.G_1darray))
        self.gam[0] = 1 / gamsum
        # TODO: HIER IST GLAUB ICH EIN SCHREIBFEHLER IM PAPER, ES MueSSTE MINUS C HEIssEN DA DIE DIMENSIONEN JA GARNICHT
        # TODO: STIMMEN KoeNNEN!!!
        self.gam[1] = 1 / ((0.5 * np.linalg.norm(np.dot(self.S, self.G) - self.C))**2)

        self.gam[2] = 1 / ((0.5 * np.dot(self.G.T, self.smoothie(self.G)) + 0.5 * np.dot(self.G.T, self.G)))

    # Calculate weights 1-3 according to p.2207
    def deltaWeights(self, n):
        _, g_delta = self.delta()
        if n == 1:
            return self.func1(g_delta) - self.func1(self.G)
        elif n == 2:
            return self.func2(g_delta) - self.func2(self.G)
        elif n == 3:
            return self.func3(g_delta) - self.func3(self.G)


    # Update weights according to the updatestep given on p.2207
    def updateWeights(self):
        for i in range(3):
            weightsum = 0
            for k in range(3):
                weightsum += (self.deltaWeights(1) / self.deltaWeights(k+1))
            self.w[i] = (self.deltaWeights(1) / self.deltaWeights(i+1))/weightsum


    # Calculate image vector according to equation 36
    def updateImage(self):
        _, self.G = self.delta()


    # Aktivierungsfunktion nach S.2204 (23)
    # Spratte: Ich glaube, das Paper hat hier einen Fehler (bzw. es fehlt
    # Information). Der Sinn einer Aktivierungsfunktion ist doch, den Wert
    # zwischen 0 und 1 zu halten. Die angegebene Funktion tut dies aber nicht.
    # Ich glaube es muesste heissen:
    #
    #               / 0               if      beta u_j + xi <= 0
    #     f(u_j) = <  beta u_j + xi   if 0 <  beta u_j + xi < 1
    #               \ 1               if 1 <= beta u_j + xi
    #
    # Das stimmt mit der Definition aus dem Paper ueberein, solange beta und xi
    # in einem bestimmten Intervall liegen (genauer: solange beta = 1).
    # Sollte meine Vermutung richtig sein, ist folgende Implementierung ein
    # bisschen schneller:
    def activation(self, u):
        v = self.beta * u + self.xi
        v[v<0] = 0
        v[v>1] = 1
        return v


    # Calculate the mean square error of the image vector
    def calcError(self):
        _, g_delta = self.delta()
        return np.linalg.norm(g_delta - self.G)**2


    # Run an updating step according to p.2207
    def updatingStep(self):
        self.calcGamma()
        self.updateWeights()
        self.updateImage()


    # Calculate the image vector G from the given capacity values C
    def calc_G(self, C):
        self.C = C
        self.G = self.initializeG(C)
        self.G_1darray = np.reshape(self.G, (len(self.G)))
        self.updatingStep()
        Error = self.calcError()
        if len(np.where(Error > 1e-4)[0]) == 0:
            return self.delta()
        else:
            self.timestep()



if __name__ == '__main__':
    ImageSize = (91, 91)
    SMatrixPfad = '../Daten/S_matrix_quadratisch.mat'
    S = importMatFile(SMatrixPfad)
    S[S==0] = 1

    CMatrixPfad = '../Daten/3_Kreise.txt'
    C = importCMatrix(CMatrixPfad)

    Model = InitModel(C, S, ImageSize)
    _, ImageSolution = Model.calc_G(C)
