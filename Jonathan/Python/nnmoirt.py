import numpy as np

# Implementation of the NNMOIRT algorithm as described in
#   Neural network based multi criterion optimization image reconstruction
#   technique for imaging two- and three-phase flow systems using electrical
#   capacitance tomography
# by W. Warsito and L.-S. Fan
# Published in Measurement Science and Technology (vol. 12), 6. Nov. 2001

# naming convention of this file:
# variables:
#   single word: TeX equivalent (so $S$ is S, $\alpha$ is alpha)
#   multiple words: CamelCases
# functions:
#   single word: TeX equivalent
#   multiple words: under_scores

class NNMOIRT:# {{{
    # doc string {{{
    """ Initialize an NNMOIRT model
        Input data:
            S (NxM np.ndarray): Sensitiviy matrix with N the size of the
                capacitance vectors and M the size of the image vector
            ImageSize (2x1 tuple): The size of the created image in HxW
            alpha_0 (float): alpha_0 as in eq. (29) (alpha_0>0)
            zeta (float): zeta as in eq. (29) (zeta>0)
            OmegaInitial (3x1 tuple): the initial values for omega_i
            activation (string): Choose the used activation function from:
                'linear': activation according to eq. (23) corrected
                'verbatim': activation according to eq. (23) uncorrected
                'sigmoid': activation according to eq. (22)
                'double step': activation according to eq. (37-39)
            beta (float or 2x1 tuple): factor in the activation functions, tuple
                only for activation='double step'
            xi (float or 2x1 tuple): factor in the activation functions, tuple
                only for activation='double step'
            SmoothingWeights (3x1 tuple): The weights used to smooth the image
                vector. 1st element is self, 2nd is for edge-neighbours, 3rd for
                vertix-neighbours.
            tau (float): time constant R_{0}C_{0} for eq. (31)
            deltaT (float): time step
            eta (float): factor for eq. (29) (eta>0)
            ActivationMin (float): you can force the values of the activation
                functions to be in bounds, this is the lower bound
            ActivationMax (float): you can force the values of the activation
                functions to be in bounds, this is the upper bound
            ActivationOffsetted (Boolean): if True the whole activation will be
                offsetted by ActivationMin. Only supported for
                activation='linear'.
    """    
    # }}}
    # init function {{{
    def __init__ (self, S, ImageSize
            ,alpha_0 = 7                        # according to paper
            ,zeta = 5                           # according to paper
            ,OmegaInitial = (1/3,1/3,1/3)       # according to paper
            ,activation = 'linear'              # most reasonable
            ,beta = 2                           # according to paper
            ,xi = 0.1                           # arbitrary
            ,SmoothingWeights = (1, -1/8, -1/8) # according to paper
            ,tau = 1                            # according to paper
            ,deltaT = 0.01                      # according to paper
            ,eta = 1                            # arbitrary
            ,ActivationMin = 1e-50              # arbitrary but reasonable
            ,ActivationMax = 1                  # according to paper
            ,ActivationOffsetted = False        # arbitrary
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
        self.OmegaInitial = OmegaInitial
        self.omega = OmegaInitial
        if len(SmoothingWeights) != 3:
            raise ValueError("SmoothingWeights must be of length 3")
        self.SmoothingWeights = SmoothingWeights
        self.tau = tau
        self.deltaT = deltaT
        self.f = ( self.f1, self.f2, self.f3 )
        # for double step
        self.rho = (1,1)
        self.ActivationMin = ActivationMin
        self.ActivationMax = ActivationMax
        self.ActivationOffsetted = ActivationOffsetted
        if ActivationOffsetted:
            if activation != 'linear':
                raise ValueError(
                        "Offsetted Activation is only supported with the "
                        "'activation' key set to 'linear'.")
        if activation == 'linear':
            if ActivationOffsetted:
                self.ActivationNewMax = ActivationMax + ActivationMin
                self.activation = self.activation_linear_offset
                self.reverse_activation = self.reverse_activation_linear_offset
            else:
                self.activation = self.activation_linear
                self.reverse_activation = self.reverse_activation_linear
        elif activation == 'sigmoid':
            self.activation = self.activation_sigmoid
            self.reverse_activation = self.reverse_activation_sigmoid
        elif activation == 'double step':
            self.activation = self.activation_double_step
            self.reverse_activation = self.reverse_activation_double_step
        elif activation == 'verbatim':
            self.activation = self.activation_linear_verbatim
            self.reverse_activation = self.reverse_activation_linear
        else:
            raise ValueError(
                    "The 'activation' key only accepts one of the four "
                    "values:\n"
                    "'linear', 'sigmoid', 'double step', 'verbatim'")
            self.activation = self.activation_linear
            self.reverse_activation = self.reverse_activation_linear
        if activation == 'double step':
            self._assert_subscriptable(beta,2,'beta')
            self._assert_subscriptable(xi,2,'xi')
        else:
            self._assert_scalar(beta,'beta')
            self._assert_scalar(xi,'xi')
    # }}}

    # assertions {{{
    # assert something is subscriptable and one-dimensional {{{
    def _assert_subscriptable(self,obj,size,name):
        if np.size(obj) < size:
            raise ValueError(
                    "'%s' is not subscriptable or isn't big enough.\n"
                    "Should be subscriptable and have size %d."%(name,size))
        if len(np.shape(obj)) != 1:
            raise ValueError("'%s' is not one-dimensional."%(name))
    # }}}
    # assert something is a scalar {{{
    def _assert_scalar(self,obj,name):
        if np.size(obj) != 1 or len(np.shape(obj)) != 0:
            raise ValueError("'%s' is not a single number"%(name))
    # }}}
    # }}}

    # activation functions {{{
    # Eq. (23)
    def activation_linear(self, u):
        v = self.beta * u + self.xi
        v[v<self.ActivationMin] = self.ActivationMin
        v[v>self.ActivationMax] = self.ActivationMax
        return v
    def reverse_activation_linear(self, v):
        return (v - self.xi) / self.beta
    def activation_linear_verbatim(self, u):
        v = self.beta * u + self.xi
        v[u<=-self.xi/self.beta] = self.ActivationMin
        v[self.ActivationMax-self.xi/self.beta<=u] = self.ActivationMax
        return v
    # Eq. (23) with a small offset of ActivationMin:
    def activation_linear_offset(self, u):
        v = self.beta * u + self.xi + self.ActivationMin
        v[v<self.ActivationMin] = self.ActivationMin
        v[v>self.ActivationNewMax] = self.ActivationNewMax
        return v
    def reverse_activation_linear_offset(self, v):
        return (v - self.xi - self.ActivationMin) / self.beta
    # Eq. (22)
    def activation_sigmoid(self, u):
        return 1 / ( 1 + np.exp(-self.beta * u) )
    def reverse_activation_sigmoid(self, v):
        return - np.log( 1/v - 1 ) / self.beta
    # Eq. (37-39)
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
    # Equations as in "2. Update step" on page 2206
    # left out the factors 1/2 since they're cancelled down anyway in f_i
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
        v[v<self.ActivationMin] = self.ActivationMin
        v[v>self.ActivationMax] = self.ActivationMax
        if self.ActivationOffsetted: v += self.ActivationMin
        return (self.reverse_activation(v), v)
    # }}}

    # f_i functions {{{
    # left out the factors 1/2 since they're cancelled down anyway with gamma_i
    # Eq. (14)
    def f1(self, G, GSmoothed, C):
        return self.gamma[0] * np.dot(G, np.log(G))
    # Eq. (15)
    def f2(self, G, GSmoothed, C):
        tmp = np.dot(self.S, G) - C
        return self.gamma[1] * np.dot(tmp, tmp)
    # Eq. (16)
    def f3(self, G, GSmoothed, C):
        return self.gamma[2] * ( np.dot(G, GSmoothed) + np.dot(G, G) )
    # }}}

    # calculate omega_i {{{
    # Equations as in "2. Update step" on page 2206
    def calc_omega(self, GFuture, GFutureSmoothed, C):
        # Since $\gamma_i != \frac{1}{f_i(G)|_{\gamma_i=1}}$ one can subsitute
        # $f_i(G)$ with 1
        DeltaOmega = [
                self.f[i](GFuture, GFutureSmoothed, C) - 1 for i in range(3)
                ]
        SumDeltaRel = 1 + \
                DeltaOmega[0] * ( 1 / DeltaOmega[1] + 1 / DeltaOmega[2] )
        return [
                ( 1 if i == 0 else DeltaOmega[0] / DeltaOmega[i] ) / SumDeltaRel
                for i in range(3)
               ]
    # }}}

    # function to smooth G returning the result of X*G {{{
    # Implements X*G with X as in eq. (17)
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

    # time dependent alpha {{{
    # Eq. (29)
    def alpha(self):
        return self.alpha_0 + self.zeta * np.exp(-self.eta * self.time)
    # }}}

    # u derivate {{{
    # Eq. (31)
    def u_deriv(self, u, G, GSmoothed, C):
        # Eq. (28)
        z = np.dot(self.S, G) - C
        # Eq. (27)
        deltaZ = self.alpha() * z
        deltaZ[z<0] = 0
        return - u / self.tau - (
                self.omega[0] * self.gamma[0] * ( 1 + np.log(G) )
                + self.omega[1] * self.gamma[1] * np.dot(self.S.T, z)
                + self.omega[2] * self.gamma[2] * (GSmoothed + G)
                + np.dot(self.S.T, deltaZ)
                )
    # }}}

    # update G {{{
    # Eq. (33) and (34)
    def update_G(self, u, G, GSmoothed, C):
        u += self.deltaT * self.u_deriv(u, G, GSmoothed, C)
        return (u, self.activation(u))
    # }}}

    # calc_G function {{{
    def calc_G(self, C, InitZeros = True, MaxIterations = int(1e4),
            residuum = 1e-4, ResiduumElementwise = False):
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
        self.omega = self.OmegaInitial
        for i in range(MaxIterations):
            print('Iteration', i)
            self.gamma = self.calc_gamma(G, GSmoothed, C)
            u, GFuture = self.update_G(u, G, GSmoothed, C)
            deltaG = GFuture - G
            if ResiduumElementwise:
                if len(np.where(deltaG**2 > residuum)[0]) == 0: break
            elif np.dot(deltaG, deltaG) <= residuum: break
            GSmoothed = self.smooth_G(GFuture)
            self.omega = self.calc_omega(GFuture, GSmoothed, C)
            G = GFuture
            self.time += self.deltaT
        if i == MaxIterations:
            print('Stopped after %d iterations!'%(MaxIterations))
        if self.ActivationOffsetted: GFuture -= self.ActivationMin
        return GFuture
    # }}}
# }}}

# vim: fdm=marker fmr={{{,}}}
