# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:01:07 2024

@author: Samuel García & Hugo Perlaza
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from tqdm import tqdm


N_tiempo = 101
N_x = 6
N_y = 6

x_vals = np.linspace(0, 1, N_x)
y_vals = np.linspace(0, 1, N_y)
tiempo_vals = np.linspace(0, 1, N_tiempo)


delta_x = x_vals[1] - x_vals[0]
delta_y = y_vals[1] - y_vals[0]
delta_t = tiempo_vals[1] - tiempo_vals[0]


alfa = beta = 1
k_val = alfa

lambda_coef = (k_val * delta_t) / (delta_x ** 2)
mu_coef = (k_val * delta_t) / (delta_y ** 2)


def temperatura_inicial(x, y):
    return np.sin(np.pi * (x + y))


def inicializar_temperatura():
    T_vals = np.zeros((N_tiempo, N_x, N_y))
    

    for i in range(N_x):
        for j in range(N_y):
            T_vals[0, i, j] = temperatura_inicial(x_vals[i], y_vals[j])
    

    T_vals[:, 0, :] = np.exp(-2 * np.pi ** 2 * tiempo_vals)[:, None] * np.sin(np.pi * y_vals)  
    T_vals[:, :, 0] = np.exp(-2 * np.pi ** 2 * tiempo_vals)[:, None] * np.sin(np.pi * x_vals)  
    T_vals[:, -1, :] = np.exp(-2 * np.pi ** 2 * tiempo_vals)[:, None] * np.sin(np.pi * (1 + y_vals)) 
    T_vals[:, :, -1] = np.exp(-2 * np.pi ** 2 * tiempo_vals)[:, None] * np.sin(np.pi * (1 + x_vals))  
    
    return T_vals

temperatura_vals = inicializar_temperatura()


def obtener_solucion():
    for l in tqdm(range(1, len(tiempo_vals))):
        temperatura_vals[l, 1:-1, 1:-1] = (1 - 2 * lambda_coef - 2 * mu_coef) * temperatura_vals[l - 1, 1:-1, 1:-1] + \
                                           lambda_coef * (temperatura_vals[l - 1, 2:, 1:-1] + temperatura_vals[l - 1, :-2, 1:-1]) + \
                                           mu_coef * (temperatura_vals[l - 1, 1:-1, 2:] + temperatura_vals[l - 1, 1:-1, :-2])

obtener_solucion()


fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

def inicializar_grafico():
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-1, 1)

inicializar_grafico()
X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
surface = [ax.plot_surface(X_mesh, Y_mesh, temperatura_vals[0, :, :], cmap='plasma')]

def actualizar_grafico(i, surface):
    surface[0].remove()
    surface[0] = ax.plot_surface(X_mesh, Y_mesh, temperatura_vals[i, :, :], cmap='plasma')

animacion_grafico = animation.FuncAnimation(fig, actualizar_grafico, frames=len(tiempo_vals), fargs=(surface,), interval=50, blit=False)
plt.show()

