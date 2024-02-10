# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:13:58 2024

@author: Samuel Garcia & Hugo Perlaza
"""

import numpy as np
import matplotlib.pyplot as plt

def Ecuacion(x, y):
    return (x**4 * y**2 - 2 * x**2 * y - 1) / x**3

def Integrador(f, x0, y0, h, x_end):
    x_values = np.arange(x0, x_end + h, h)
    y_values = np.zeros_like(x_values)
    y_values[0] = y0

    for i in range(1, len(x_values)):
        k1 = h * f(x_values[i-1], y_values[i-1])
        k2 = h * f(x_values[i-1] + h/2, y_values[i-1] + k1/2)
        k3 = h * f(x_values[i-1] + h/2, y_values[i-1] + k2/2)
        k4 = h * f(x_values[i-1] + h, y_values[i-1] + k3)

        y_values[i] = y_values[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x_values, y_values

# Condiciones inciales
x0 = np.sqrt(2)
y0 = 0
h = 0.01
x_end = 20

# Solución numérica
x_values, y_values = Integrador(Ecuacion, x0, y0, h, x_end)


plt.plot(x_values, y_values, label='Solución Numérica')
plt.plot(x_values, x_values**(-2), label='Solución Particular $y_1=x^{-2}$', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solución numérica')
plt.legend()
plt.show()
