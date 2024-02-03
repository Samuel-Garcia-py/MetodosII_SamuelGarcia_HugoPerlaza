# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:12:08 2024

@author: Samuel García y Hugo Perlaza
"""

import numpy as np

def f(x):
    return np.sin(x)

def primera_derivada_conocida(x):
    return np.cos(x)

def segunda_derivada_comocida(x):
    return -np.sin(x)

# Operador 1:
def operador_1(x, h):
    return (-f(x + 2*h) + 4*f(x + h) - 3*f(x)) / (2 * h)

def operador_2(x,h):
    return (f(x + h) - 2*f(x) + f(x-h)) / (h**2)

# Rango de valores de x
x_values = np.linspace(0, 2*np.pi, 100)

# Tamaños de paso h
TamañoDePaso= [0.1, 0.01, 0.001]


for h in TamañoDePaso:
    print("Aproximación con h = " + str(h))
    for x in x_values:
        approx1 = operador_1(x, h)
        approx2 = operador_2(x, h)
        real1 = primera_derivada_conocida(x)
        real2 = segunda_derivada_comocida(x)
        error1 = abs(approx1 - real1)
        error2 = abs(approx2 - real2)
        print("x = " + str(round(x, 2)) + ", Aproximación 1 = " + str(round(approx1, 6)) + ", Derivada real = " + str(round(real1, 6)) + ", Error = " + str(round(error1, 6)))
        print("x = " + str(round(x, 2)) + ", Aproximación 2 = " + str(round(approx2, 6)) + ", Derivada real = " + str(round(real2, 6)) + ", Error = " + str(round(error2, 6)))
    print()

