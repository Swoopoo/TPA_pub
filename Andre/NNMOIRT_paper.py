import numpy as np

# Basiert auf Paper: 'Neural network based multi-criterion
# optimization image reconstruction
# technique for imaging two- and
# three-phase flow systems using electrical
# capacitance tomography'
# -----------------------------------------------------
# von W. Warsito und L-S Fan
# -----------------------------------------------------
# -----------------------------------------------------
# STAND: 18.05.2018

# TODO: Geschwindigkeitsoptimierungen:
#  - len(np.array) ist langsamer als np.array.size bei 1-D arrays

class InitModel:

    def __init__(self, G, C, S):
        self.t = 0
        self.deltat = 0.01
        self.tau = 1 # = R_0 C_0
        self.alpha_0 = 7
        self.beta = 2
        self.xi = 5
        self.gam = np.zeros(3)
        self.u = np.zeros(len(G))
        self.u_deltat = np.zeros(len(G))
        self.v = self.activation(self.u)
        self.G = G
        self.G_deltat = np.zeros(len(G))
        self.C = C
        self.S = S
        self.z, self.deltaz = self.rel()
        self.X = self.smoothMat()
        self.XG = self.smoothie(G)
        self.smoothing_weights = (1, -1/8, -1/8)
        self.w = np.array([1/3, 1/3, 1/3])

    # Differential von u nach S. 2205 (31)
    # Ist self.gam hier wirklich das gleiche wie in den anderen Funktionen? Im
    # Text steht auf S. 2204 das gamma = gamma / C_0 substituiert wird.
    def derivu(self):
        u_deriv = -(self.u/self.tau) - (
                  self.w[0] * self.gam[0] * (1 + np.log(self.G))
                + self.w[1] * self.gam[1] * np.dot(self.S.T, self.z)
                + self.w[2] * self.gam[2] * (np.dot(self.X, self.G) + self.G)
                + np.dot(self.S.T, self.deltaz)
                )
        return u_deriv

    # Alpha nach 2204 (29)
    def alpha(self, t):
        # Spratte: kleiner Fehler, der komische griechische Buchstabe ist ein
        # zeta: alpha_0, zeta, eta >= 0
        #return self.alpha_0 + self.zeta * np.exp(-self.eta * t)
        return self.alpha_0 + self.xi * np.exp(-t)

    # G(t+deltat) bzw. u(t+deltat) nach S. 2205 (36) bzw. (33)
    def delta(self):
        u_deltat = self.u + self.derivu() * self.deltat
        G_deltat = self.G + self.beta * self.derivu() * self.deltat
        return u_deltat, G_deltat

    # Zeitschritt machen
    def timestep(self):
        self.t += self.deltat
        self.u, self.G = self.delta()

    # z und deltaz nach S. 2204 (28) und (27)
    # Spratte: Müsste ich mir zwar nochmal ansehen, aber das müsste auch ohne
    # doppelte for-Schleife implementierbar sein (for-Schleifen sind teuer in
    # Python, die numpy-Routinen sind wesentlich schneller).
    # z müsste doch eigentlich
    #    z = np.dot(self.S, self.G) - self.C
    # sein. Dementsprechend wäre die Funktion hier dann:
    #def rel(self):
        #z = np.dot(self.S, self.G) - self.C
        #deltaz = self.alpha(self.t) * z
        #deltaz[np.where(deltaz < 0)] = 0
    # Ich bin mir nicht sicher, ob die Implementierung oben wirklich optimal
    # ist oder ob es schneller ginge, nähme man eine for-Schleife für die
    # Berechnung von deltaz.
    def rel(self): # delta(z) Berechnung
        z = np.zeros(len(self.C))
        deltaz = np.zeros(len(self.C))
        for i in range(len(self.C)):
            zsum = 0
            for j in range(len(self.G)):
                zsum += self.S[i, j] * self.G[j] - self.C[i]
            z[i] = zsum
            if z[i] > 0: deltaz[i] = self.alpha(self.t) * z[i]
        return z, deltaz

    # Objektfunktion f1 nach S. 2203 (14)
    def func1(self, G):
        # Entropie Funktion des Bildes - lässt Aussage über die 'global smoothness' des Bildvektors treffen (14)
        # G = Bildvektor
        # len(G) = Anzahl an Neuronen
        # G(j) = Wert des j-ten Neurons
        # ga = normalisierte Konstante 0 <= ga1 <= 1

        f1 = self.gam[0] * np.dot(G,np.log(G))
        return f1

    # Objektfunktion f2 nach S. 2203 (15)
    def func2(self, G):
        # zurückrechnen auf Kapazitäten (15)
        # S = Sensitivitätsmatrix
        # G = Bildvektor
        # C = Gemessene Kapazitäten
        # ga2 = normalisierte Konstante 0 <= ga2 <= 1
        #
        # Anmerkungen:
        #  - Warum ist C in self, aber G nicht? Müsste C nicht auch extern sein?
        #    Immerhin ändert sich C wie G mit jeder Messung, während S statisch
        #    ist.
        err = np.zeros((len(G),len(G)))
        errsq = np.zeros((len(G), len(G)))
        for i in range(len(G)):
            for j in range(len(G)):
                err += self.S[i, j] * G[j] - self.C[i]
            errsq += err ** 2
        f2 = 0.5 * self.gam[1] * errsq
        return f2
        # Alternative Implementierung von Spratte:
        #   Datentyp abhängig:
        #     self.S : np.array
        #     self.C : np.array
        #     G      : np.array
        # entweder (bei kleinen Arrays schneller)
        f2 = .5 * self.gam[1] * np.sum(np.square(self.S.dot(G) - self.C))
        # oder (bei großen Arrays schneller)
        f2 = .5 * self.gam[1] * np.sum(np.square(self.S @ G - self.C))
        #   Matrix soll overhead haben gegenüber array
        #   (https://stackoverflow.com/a/3892639/6783373) -> array ist besser
        #     self.S : np.matrix
        #     self.C : np.matrix
        #     G      : np.matrix
        # f2 = .5 * self.gam[1] * np.sum(np.square(self.S * G - C))

    # Objektfunktion f3 nach S. 2203 (16)
    def func3 (self, G):  # sum of the non-uniformity and peakedness functions of an image (16) - minimiert um locale
        #  'smoothness' und kleine 'peakedness' zu gewährleisten
        # G = Image Vektor
        # X = N x N non-uniformity Matrix, 'smoothing' Matrix, für Wert[i,k] gilt (1 <= j <= N):
        #           x1  wenn    k = j
        #           x2  wenn    k E E[j]
        #           x3  wenn    k E V[j]
        #           0   wenn    sonst
        #       E[j] Indexe der Pixel, die genau EINE Kante mit dem Pixel j gemeinsam haben
        #       V[j] Indexe der Pixel, die genau EINEN Knoten mit dem Pixel j gemeinsam haben
        #       x1, x2, x3 == smoothing weights == (1, -1/8, -1/8) für die Standard-Smoothing Matrix
        f3 = 0.5 * self.gam[2] * (np.dot(np.dot(G.T, self.X), G) + np.dot(G.T,G))
        return f3

    # Spratte: Lösung für Berechnung der smoothMat (S.2202):
    # Unterstellt: Rechteckiges Bild mit Breite self.G_width, G[0] in linker
    # oberer Ecke, G[1] der zweite Pixel von links in der obersten Reihe
    # (Reihenfolge eigentlich nicht wirklich wichtig, bloß G[0] in einer Ecke
    # und self.G_width die Breite in die erste Richtung -- also
    # G[self.G_width-1] der letzte Pixel in der ersten Reihe/Spalte). Die
    # Funktion skaliert allerdings relativ schlecht (O(n) = n**2)
    # Spratte: Für große Matrizen wird extrem viel Speicher benötigt. Auf meinem
    # Laptop ist bei 200x200 Pixeln Bildgröße schluss. Eventuell müssen wir auf
    # langsamere aber speicherfreundlichere Berechnungen umsteigen -- oder aber
    # ich schreibe doch noch ein Modul in C.
    # Die Speichermenge wechst in 4ter Potenz mit der Bildbreite, bzw.
    # quadtratisch zur Pixelzahl, wobei ein 1x1 np.array 8 byte benötigt.
    #def smoothMat2(size,width):
        #X = np.zeros((size,size))
        #for i in range(size):
            #X[i][i] = 1 #x1
            #for j in range(i+1,size):
                #i_row = i // width # Python 3 Syntax für Integerdivision
                #i_col = i %  width
                #j_row = j // width # Python 3 Syntax für Integerdivision
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
        #i_row = i // self.G_width # Python 3 Syntax für Integerdivision
        #i_col = i %  self.G_width
        #j_row = j // self.G_width # Python 3 Syntax für Integerdivision
        #j_col = j %  self.G_width
        #if i_row == j_row          and abs(i_col - j_col) == 1 : return x2
        #if abs(i_row - j_row) == 1 and i_col == j_col          : return x2
        #if abs(i_row - j_row) == 1 and abs(i_col - j_col) == 1 : return x3
        #return 0
    # Berechnet das Matrixprodukt X*G
    def smoothie(self, G):
        XG = self.smoothing_weights[0] * G.copy()
        for i in range(XG.size):
            E, V = get_neighbours(i) # returns 2 lists of indices
            XG[i] +=   self.smoothing_weights[1] * np.sum(G[E]) \
                     + self.smoothing_weigths[2] * np.sum(G[V])
        return XG
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

    # Smoothing Matrix (S. 2202)
    # Spratte: Habe einen kurzen Test gemacht und es scheint, deine Funktion
    # schmeißt Errors um sich. Ich habe getestet mit:
    #IssueCount = 0
    #for i in range(201):
        #try:
            #X = smoothMat(i**2)
        #except:
            #print("Issue with", i)
            #IssueCount += 1
    #print("Issues in ",IssueCount," iterations")
    # und nur für 4 (also smoothMat(16)) funktioniert die Funktion. Das Argument
    # war in diesem Fall len(self.G) (in meiner Testumgebung habe ich
    # dementsprechend alle Aufrufe ersetzt).
    def smoothMat(self):
        X = np.zeros((len(self.G), len(self.G)))
        ind = np.array(range(0, len(self.G))).reshape(int(np.sqrt(len(self.G))), int(np.sqrt(len(self.G))))
        x1, x2, x3 = 1, -1 / 8, -1 / 8
        E = np.zeros((1, int(np.sqrt(len(self.G)))))
        V = np.zeros((1, int(np.sqrt(len(self.G)))))

        for i in range(int(np.sqrt(len(self.G)))):
            for j in range(int(np.sqrt(len(self.G)))):
                E_temp, V_temp = self.neighbours((i, j))
                for k in range(len(E_temp)):
                    # if
                    if E_temp[k] is not np.NaN:
                        E_temp[k] = ind[E_temp[k][0], E_temp[k][1]]
                    if V_temp[k] is not np.NaN:
                        V_temp[k] = ind[V_temp[k][0], V_temp[k][1]]
                if E_temp.shape == (int(np.sqrt(len(self.G))), 2):
                    E_temp = np.array([E_temp[k][0] for k in range(4)])
                if V_temp.shape == (int(np.sqrt(len(self.G))), 2):
                    V_temp = np.array([V_temp[k][0] for k in range(4)])
                E = np.vstack((E, E_temp))
                V = np.vstack((V, V_temp))
        E = np.delete(E, 0, 0)
        V = np.delete(V, 0, 0)
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                if k in E[j]:
                    X[j, k] = x2
                elif k in V[j]:
                    X[j, k] = x3
                else:
                    X[j, k] = 0
        np.fill_diagonal(X, x1)
        return X

    # Nachbar Indizes für Smoothing Matrix
    def neighbours(self, a):
        o, u, l, r = (a[0] - 1, a[1]), (a[0] + 1, a[1]), (a[0], a[1] - 1), (a[0], a[1] + 1)
        lo, ro, lu, ru = (a[0] - 1, a[1] - 1), (a[0] - 1, a[1] + 1), (a[0] + 1, a[1] - 1), (
            a[0] + 1, a[1] + 1)
        E = np.stack((o, u, l, r), axis=0)
        V = np.stack((lo, ro, lu, ru), axis=0)
        b = []
        for n in range(E.shape[0]):
            if not all(0 <= k < int(np.sqrt(len(self.G))) for k in E[n, :]):
                b.append(n)
        # E = np.delete(E, (a), axis=0)
        E = list(E)
        for i in b:
            E[i] = np.NaN

        b = []
        for n in range(V.shape[0]):
            if not all(0 <= k < int(np.sqrt(len(self.G))) for k in V[n, :]):
                b.append(n)
        # V = np.delete(V, (a), axis=0)
        V = list(V)
        for i in b:
            V[i] = np.NaN
        return np.array(E, object), np.array(V, object)

    # Gamma-Konstanten für die Objektfunktionen nach S. 2206
    def calcGamma(self):
        gamsum = 0
        for j in range(len(self.G)):
            gamsum += self.G[j] * np.log(self.G[j])
        self.gam[0] = 1 / gamsum

        self.gam[1] = 1 / (0.5 * np.abs(np.dot(self.S, self.G) - self.G))**2

        self.gam[2] = 1 / (0.5 * np.dot(np.dot(self.G.T, self.X), self.G) + 0.5 * np.dot(self.G.T, self.G))

    # Gewichtungsänderung bei Iterationsschritt berechnen nach S. 2206
    def deltaWeights(self, n):
        _, g_delta = self.delta()
        if n == 1:
            return self.func1(g_delta) - self.func1(self.G)
        elif n == 2:
            return self.func2(g_delta) - self.func2(self.G)
        elif n == 3:
            return self.func3(g_delta) - self.func3(self.G)

    # Gewichtungen updaten nach S. 2206
    def updateWeights(self):
        for i in range(3):
            weightsum = 0
            for k in range(1,4):
                weightsum += (self.deltaWeights(1) / self.deltaWeights(k+1))
            self.w[i] = (self.deltaWeights(1) / self.deltaWeights(i+1))/weightsum

    # Bildvektor Updaten
    def updateImage(self):
        _, self.G = self.delta()

    # Aktivierungsfunktion nach S.2204 (23)
    # Spratte: Ich glaube, das Paper hat hier einen Fehler (bzw. es fehlt
    # Information). Der Sinn einer Aktivierungsfunktion ist doch, den Wert
    # zwischen 0 und 1 zu halten. Die angegebene Funktion tut dies aber nicht.
    # Ich glaube es müsste heißen:
    #
    #               / 0               if      beta u_j + xi <= 0
    #     f(u_j) = <  beta u_j + xi   if 0 <  beta u_j + xi < 1
    #               \ 1               if 1 <= beta u_j + xi
    #
    # Das stimmt mit der Definition aus dem Paper überein, solange beta und xi
    # in einem bestimmten Intervall liegen (genauer: solange beta = 1).
    # Sollte meine Vermutung richtig sein, ist folgende Implementierung ein
    # bisschen schneller:
    #def activation(self, u):
        #v = self.beta * u + self.xi
        #for j in range(len(v)):
            #if v[j] < 0: v[j] = 0
            #elif v[j] > 1: v[j] = 1
        #return v
    def activation(self, u):
        v = np.zeros(len(u))
        for j in range(len(u)):
            if u[j] <= -self.xi / self.beta:
                v[j] = 0
            elif u[j] >= 1 - (self.xi / self.beta):
                v[j] = self.beta * u[j] + self.xi
            else:
                v[j] = 1
        return v

    # Fehlerfunktion nach S.2207
    def calcError(self):
        _, g_delta = self.delta()
        return np.abs(g_delta - self.G)**2


# G und C als .shape = (bla,) Vektoren übergeben!
x = InitModel(np.random.randint(1, 9, size=16)/100, np.random.randint(1, 9, size=12)/1000,
              np.random.randint(1, 4, size=(12, 16)))

print(x.G)
print(x.X)
