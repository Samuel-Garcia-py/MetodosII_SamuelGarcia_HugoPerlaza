# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 06:19:51 2024

@author: Samuel García & Hugo Perlaza
"""

# Algoritmo de Verlet



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x0 = 1.0
v0 = 1.0
w = np.pi


hEstable = 0.01
hInestable = 0.1
pasos = 1000

def verlet(x0, v0, w, h, pasos):
    posiciones = np.zeros(pasos)
    Velocidades = np.zeros(pasos)

    
    posiciones[0] = x0
    Velocidades[0] = v0

    
    R = h**2 * w**2

    for i in range(1, pasos):
        posiciones[i] = 2 * posiciones[i - 1] - posiciones[i - 2] - R * posiciones[i - 1]
        Velocidades[i] = (posiciones[i] - posiciones[i - 2]) / (2 * h)

    return posiciones



# Simulación para la región estable
positions_stable = verlet(x0, v0, w, hEstable, pasos)

# Simulación para la región inestable
positions_unstable = verlet(x0, v0, w, hInestable, pasos)

# Animación
fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.plot(positions_stable[:frame], label='Región Estable', color='green')
    ax.plot(positions_unstable[:frame], label='Región Inestable', color='red')
    ax.legend()
    ax.set_title('Verlet para Oscilador')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Posición')

ani = FuncAnimation(fig, update, frames=pasos, interval=50, repeat=False)
plt.show()
