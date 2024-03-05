# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:52:21 2024

@author: samue
"""

import numpy as np
import matplotlib.pyplot as plt

#a

''' La demostración se adjunta en pdf en la carpeta'''

# b
def calcular_Rn_Sn(x, E):
    m = 1
    omega = 1
    V = 0.5 * m * omega**2 * x**2
    Rn = 2 * (E - V)
    Sn = np.zeros_like(x)
    return Rn, Sn

# c
N = 1000
x = np.linspace(-5, 5, N)
h = x[1] - x[0]

# d
def potencial(x):
    return 0.5 * x**2

# e
def metodo_numerov(x, E):
    psi = np.zeros_like(x)
    psi[0] = 0.01
    psi[1] = 0.01

    Rn, _ = calcular_Rn_Sn(x, E)

    for i in range(2, N):
        psi[i] = (2 * (1 - (5 * (h**2) / 12) * Rn[i]) * psi[i-1] -
                  (1 + (h**2) / 12 * Rn[i-1]) * psi[i-2])
        

        max_psi = np.max(np.abs(psi))
        if max_psi > 1:
            psi /= max_psi


    psi /= np.max(np.abs(psi))

    return psi

# f
def encontrar_valores_propios():
    valores_propios = []
    dE = 0.001
    for E in np.arange(0.5, 6, dE):
        psi = metodo_numerov(x, E)
        if psi[-1] * metodo_numerov(x, E + dE)[-1] < 0:
            valores_propios.append(E)
    return valores_propios[:6]

# g
valores_propios = encontrar_valores_propios()
print("Valores propios encontrados:", valores_propios)


plt.figure(figsize=(10, 6))
for i, E in enumerate(valores_propios):
    psi = metodo_numerov(x, E)
    plt.plot(x, psi, label=f"E = {E:.2f}")
plt.title("Estados propios del operador del oscilador armónico cuántico")
plt.xlabel("x")
plt.ylabel("Psi(x)")
plt.legend()
plt.ylim(-1, 1)  
plt.yticks(np.arange(-1, 1.25, 0.25))  
plt.grid(True)
plt.show()

# Potencial Gaussiano
def potencial_gaussiano(x):
    return -10 * np.exp(-x**2 / 20)

def encontrar_valores_propios_gaussiano():
    En_gaussiano = [-9.51, -8.54, -7.62, -6.74, -5.89]
    valores_propios_gaussiano = []
    for E in En_gaussiano:
        psi = metodo_numerov(x, E)
        valores_propios_gaussiano.append((E, psi))
    return valores_propios_gaussiano

valores_propios_gaussiano = encontrar_valores_propios_gaussiano()


plt.figure(figsize=(10, 6))
for E, psi in valores_propios_gaussiano:
    plt.plot(x, psi, label=f"E = {E:.2f}")
plt.title("Estados propios del potencial Gaussiano")
plt.xlabel("x")
plt.ylabel("Psi(x)")
plt.legend()
plt.grid(True)
plt.show()

# Potencial Racional
def potencial_racional(x):
    return -4 / (1 + x**2)**2

def encontrar_valores_propios_racional():
    En_racional = [-1.478, -0.163]
    valores_propios_racional = []
    for E in En_racional:
        psi = metodo_numerov(x, E)
        valores_propios_racional.append((E, psi))
    return valores_propios_racional

valores_propios_racional = encontrar_valores_propios_racional()


plt.figure(figsize=(10, 6))
for E, psi in valores_propios_racional:
    plt.plot(x, psi, label=f"E = {E:.3f}")
plt.title("Estados propios del potencial Racional")
plt.xlabel("x")
plt.ylabel("Psi(x)")
plt.legend()
plt.grid(True)
plt.show()


