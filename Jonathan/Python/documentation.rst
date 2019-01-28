Teamprojektarbeit POSPRA
========================
-----------------
NNMOIRT in Python
-----------------

**Activation function**

.. code-block:: python

    def activation_linear(self, u):
        v = self.beta * u + self.xi
        v[v<self.ActivationMin] = self.ActivationMin
        v[v>self.ActivationMax] = self.ActivationMax
        return v

