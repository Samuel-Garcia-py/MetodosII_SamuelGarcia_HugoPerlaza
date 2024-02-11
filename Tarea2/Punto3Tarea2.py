# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:13:58 2024

@author: Samuel Garcia & Hugo Perlaza
"""

# Aquí adapto el código de RungeKutta de la clase.

import numpy as np
import matplotlib.pyplot as plt

def Ecuacion(x, y):
    return (x**4 * y**2 - 2 * x**2 * y - 1) / x**3

def Integrador(f, x0, y0, h, xf):
    x = np.arange(x0, xf + h, h)
    y = np.zeros_like(x)
    y[0] = y0

    for i in range(1, len(x)):
        k1 = h * f(x[i-1], y[i-1])
        k2 = h * f(x[i-1] + h/2, y[i-1] + k1/2)
        k3 = h * f(x[i-1] + h/2, y[i-1] + k2/2)
        k4 = h * f(x[i-1] + h, y[i-1] + k3)

        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x, y

# Condiciones inciales
x0 = np.sqrt(2)
y0 = 0
h = 0.01
x_end = 20

# Solución numérica
x, y = Integrador(Ecuacion, x0, y0, h, x_end)


plt.plot(x, y, label='Solución Numérica')
plt.plot(x, x**(-2), label='Solución Particular $y_1=x^{-2}$', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solución numérica')
plt.legend()
plt.show()
