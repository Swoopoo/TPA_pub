import numpy as np

# naming convention of this file:
# variables:
#   single word: TeX equivalent (so $S$ is S, $\alpha$ is alpha)
#   multiple words: CamelCases
# functions:
#   single word: TeX equivalent
#   multiple words: under_scores

class NNMOIRT:# {{{
    # init function {{{
    def __init__ (self, S, ImageSize
            ,alpha_0 = 7
            ,zeta = 5
            ,omega = (1/3,1/3,1/3)
            ,activation = 'linear'
            ,beta = 2
            ,xi = 0.1
            ,SmoothingWeights = (1, -1/8, -1/8)
            ,tau = 1
            ,deltaT = 0.01
            ,eta = 1
            ):
        self.S = S
        self.ImgWd = ImageSize[0]
        self.ImgHt = ImageSize[1]
        self.ImgSz = self.ImgWd * self.ImgHt
        if self.S.shape[1] != self.ImgSz:
            raise ValueError(
                "Second dimension of S matrix doesn't match image size")
        self.beta = beta
        self.xi = xi
        self.alpha_0 = alpha_0
        self.time = 0
        self.eta = eta
        self.zeta = zeta
        self.zeta = zeta
        self.omega = omega
        self.SmoothingWeights = SmoothingWeights
        self.tau = tau
        self.deltaT = deltaT
        self.f = ( self.f1, self.f2, self.f3 )
        # for double step
        self.rho = (1,1)
        if activation == 'linear':
            self.activation = self.activation_linear
            self.reverse_activation = self.reverse_activation_linear
        elif activation == 'sigmoid':
            self.activation = self.activation_sigmoid
            self.reverse_activation = self.reverse_activation_sigmoid
        elif activation == 'double step':
            self.activation = self.activation_double_step
            self.reverse_activation = self.reverse_activation_double_step
        else:
            raise ValueError(
                    "The 'activation' key only accepts one of the three keys:\n"
                    "'linear', 'sigmoid', 'double step'")
            self.activation = self.activation_linear
            self.reverse_activation = self.reverse_activation_linear
    # }}}

    # activation functions {{{
    def activation_linear(self, u):
        v = self.beta * u + self.xi
        v[v<=0] = 1e-50
        v[v>1] = 1
        return v
    def reverse_activation_linear(self, v):
        return (v - self.xi) / self.beta
    def activation_sigmoid(self, u):
        return 1 / ( 1 + np.exp(-self.beta * u) )
    def reverse_activation_sigmoid(self, v):
        return - np.log( 1/v - 1 ) / self.beta
    def activation_double_step(self, u):
        v  = self.rho[0] / ( 1 + np.exp( -self.beta[0] * ( u + self.xi[0] ) ) )
        v += self.rho[1] / ( 1 + np.exp( -self.beta[1] * ( u + self.xi[1] ) ) )
        return v
    def reverse_activation_double_step(self, v):
        # Pretty sure that is the wrong error to raise in this case...
        raise SystemError('Initialization with "activation=\'double step\'" '
                'not supported with "InitZeros=False".')
    # }}}

    # calculate gamma_i {{{
    def calc_gamma1(self, G):
        return 1 / ( np.dot(G, np.log(G)) )
    def calc_gamma2(self, G, C):
        tmp = np.dot(self.S, G) - C
        return 1 / ( np.dot(tmp,tmp) )
    def calc_gamma3(self, G, GSmoothed):
        return 1 / ( np.dot(G, GSmoothed) + np.dot(G,G) )
    def calc_gamma(self, G, GSmoothed, C):
        return (
                self.calc_gamma1(G),
                self.calc_gamma2(G, C),
                self.calc_gamma3(G, GSmoothed)
                )
    # }}}

    # init u, v functions {{{
    def init_zeros(self):
        u = np.zeros(self.ImgSz)
        return (u, self.activation(u))
    def init_SC(self,C):
        v = np.dot(self.S.T, C)
        v[v<0] = 0
        v[v<1] = 1
        return (self.reverse_activation(v), v)
    # }}}

    # f_i functions {{{
    def f1(self, G, GSmoothed, C):
        return self.gamma[0] * np.dot(G, np.log(G))
    def f2(self, G, GSmoothed, C):
        tmp = np.dot(self.S, G) - C
        return self.gamma[1] * np.dot(tmp, tmp)
    # C is not necessary but we can use it in a loop if it accepts the argument
    def f3(self, G, GSmoothed, C):
        return self.gamma[2] * ( np.dot(G, GSmoothed) + np.dot(G, G) )
    # }}}

    # calculate omega_i {{{
    def calc_omega(self, GFuture, GFutureSmoothed, C):
        # Since $\gamma_i != \frac{1}{f_i(G)|_{\gamma_i=1}}$ one can subsitute
        # $f_i(G)$ with 1
        DeltaOmega = [
                self.f[i](GFuture, GFutureSmoothed, C) - 1 for i in range(3)
                ]
        SumDeltaRel = 1 + \
                DeltaOmega[0] * ( 1 / DeltaOmega[1] + 1 / DeltaOmega[2] )
        return [ DeltaOmega[0] / SumDeltaRel / DeltaOmega[i] for i in range(3) ]
    # }}}

    # function to smooth G returning the result of X*G {{{
    def smooth_G(self, G):
        XG = self.SmoothingWeights[0] * G.copy()
        for i in range(XG.size):
            E, V = self.get_neighbours(i) # returns 2 lists of indices
            XG[i] +=   self.SmoothingWeights[1] * np.sum(G[E]) \
                     + self.SmoothingWeights[2] * np.sum(G[V])
        return XG
    # Find neighbours for the smoothie calculation (lule)
    def get_neighbours(self, i):
        E = []
        V = []
        irow = i // self.ImgWd
        icol = i - self.ImgWd * irow # faster than modulo
        if irow != 0:
            E.append(i - self.ImgWd)
            if icol != 0:
                V.append(i - self.ImgWd - 1)
            if icol != self.ImgWd - 1:
                V.append(i - self.ImgWd + 1)
        if irow != self.ImgHt - 1:
            E.append(i + self.ImgWd)
            if icol != 0:
                V.append(i + self.ImgWd - 1)
            if icol != self.ImgWd - 1:
                V.append(i + self.ImgWd + 1)
        if icol != 0:
            E.append(i - 1)
        if icol != self.ImgWd - 1:
            E.append(i + 1)
        return (E, V)
    # }}}

    # update G {{{
    def update_G(self, u, G, GSmoothed, C):
        u += self.deltaT * self.u_deriv(u, G, GSmoothed, C)
        return (u, self.activation(u))
    # }}}

    # time dependent alpha {{{
    def alpha(self):
        return self.alpha_0 + self.zeta * np.exp(-self.eta * self.time)
    # }}}

    # u derivate {{{
    def u_deriv(self, u, G, GSmoothed, C):
        z = np.dot(self.S, G) - C
        deltaZ = self.alpha() * z
        deltaZ[z<0] = 0
        return - u / self.tau - (
                self.omega[0] * self.gamma[0] * ( 1 + np.log(G) )
                + self.omega[1] * self.gamma[1] * np.dot(self.S.T, z)
                + self.omega[2] * self.gamma[2] * (GSmoothed + G)
                + np.dot(self.S.T, deltaZ)
                )
    # }}}

    # calc_G function {{{
    def calc_G(self, C, InitZeros = True, MaxIterations = int(1e8), residuum = 1e-4):
        # doc string {{{
        """ calc_G: Calculates the image vector G for the measurement C.
            Input data:
                C (np.ndarray): Capacitances measured
                InitZeros (Boolean): if true init u=0, else G = S.T*C
                MaxIterations (integer): Maximum number of iterations
                residuum (float): Residuum, if GFuture differs less than
                    residuum from G stop iterating.
            Output data:
                G (np.ndarray): the calculated image as a vector
        """
        # }}}
        ValueErrorFlag = False
        if len(C.shape) == 1:
            if C.shape[0] != self.S.shape[0]: ValueErrorFlag = True
        elif len(C.shape) == 2:
            if   C.shape[0] == 1 and C.shape[1] == self.S.shape[0]:
                C = C[0,:]
            elif C.shape[0] == self.S.shape[0] and C.shape[1] == 1:
                C = C[:,0]
            else: ValueErrorFlag = True
        else: ValueErrorFlag = True
        if ValueErrorFlag:
            raise ValueError('C has wrong dimensions.')
        if InitZeros:
            u, G = self.init_zeros()
        else:
            u, G = self.init_SC(C)
        GSmoothed = self.smooth_G(G)
        self.time = 0
        for i in range(MaxIterations):
            print('Iteration', i)
            self.gamma = self.calc_gamma(G, GSmoothed, C)
            u, GFuture = self.update_G(u, G, GSmoothed, C)
            tmp = GFuture - G
            if np.dot(tmp, tmp) <= residuum: break
            GSmoothed = self.smooth_G(GFuture)
            self.omega = self.calc_omega(GFuture, GSmoothed, C)
            G = GFuture
            self.time += self.deltaT
        if i == MaxIterations:
            print('Stopped after %d iterations!'%(MaxIterations))
        return GFuture
    # }}}
# }}}

# vim: fdm=markar fmr={{{,}}}
