# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:37:40 2024

@author: samue
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from tqdm.notebook import tqdm


N_tiempo = 500
N_x = 25
N_y = 25


x_vals = np.linspace(0, 2, N_x)
y_vals = np.linspace(0, 2, N_y)
tiempo_vals = np.linspace(0, 3, N_tiempo)


velocidad = np.sqrt(2)


delta_x, delta_y = x_vals[1] - x_vals[0], y_vals[1] - y_vals[0]
delta_t = tiempo_vals[1] - tiempo_vals[0]
lambda_coef = velocidad * delta_t / delta_x
mu_coef = velocidad * delta_t / delta_y



def inicial(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


X, Y = np.meshgrid(x_vals, y_vals)


u_vals = np.zeros((N_tiempo, N_x, N_y))
u_vals[0, :, :] = inicial(X, Y)


def obtener_solucion():
    for l in tqdm(range(1, len(tiempo_vals))):
        if l == 1:
            u_vals[l, :, :] = u_vals[l-1, :, :]
        else:
            u_vals[l, 1:-1, 1:-1] = 2*(1-lambda_coef**2-mu_coef**2)*u_vals[l-1, 1:-1, 1:-1] \
                + lambda_coef**2*(u_vals[l-1, 2:, 1:-1] + u_vals[l-1, :-2, 1:-1]) \
                + mu_coef**2*(u_vals[l-1, 1:-1, 2:] + u_vals[l-1, 1:-1, :-2]) \
                - u_vals[l-2, 1:-1, 1:-1]

obtener_solucion()


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)

def inicializar_grafico():
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(-1, 1)

inicializar_grafico()

def actualizar_grafico(i):
    ax.clear()
    inicializar_grafico()
    ax.plot_surface(X_mesh, Y_mesh, u_vals[i, :, :], cmap='viridis')
    ax.set_title(f'Tiempo = {tiempo_vals[i]:.2f}s')

animacion = animation.FuncAnimation(fig, actualizar_grafico, frames=len(tiempo_vals), interval=50)
plt.show()

Writer = animation.writers['pillow']
writer = Writer(fps=20)
