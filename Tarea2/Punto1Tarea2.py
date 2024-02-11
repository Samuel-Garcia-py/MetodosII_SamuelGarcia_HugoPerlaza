# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:36:16 2024

@author: Samuel García & Hugo Perlaza
"""

import numpy as np
import matplotlib.pyplot as plt


def R3(f, x0, y0, h, xf):
    x = np.arange(x0, xf + h, h)
    y = np.zeros_like(x)
    y[0] = y0

    for i in range(1, len(x)):
        k1 = h * f(x[i-1], y[i-1])
        k2 = h * f(x[i-1] + h/2, y[i-1] + k1/2)
        k3 = h * f(x[i-1] + 3*h/4, y[i-1] + 3*k2/4)
        y[i] = y[i-1] + (2*k1 + 3*k2 + 4*k3) / 9

    return x, y


def R4(f, x0, y0, h, xf):
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

# Ecuación diferencial de Riccati
def riccati(x, y):
    return (x**4 * y**2 - 2 * x**2 * y - 1) / x**3


x0 = np.sqrt(2)
y0 = 0
h = 0.01
xf = 5

# Solución numérica R3
x_rk3, y_rk3 = R3(riccati, x0, y0, h, xf)

# Solución numérica R4
x_rk4, y_rk4 = R4(riccati, x0, y0, h, xf)


plt.plot(x_rk3, y_rk3, label='RK3', linestyle='-')
plt.plot(x_rk4, y_rk4, label='RK4', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Estimación Runge-Kutta con Riccati')
plt.legend()
plt.show()

