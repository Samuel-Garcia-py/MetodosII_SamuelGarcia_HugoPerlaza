# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:20:54 2024

@author: Samuel García & Hugo Perlaza
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


N_t = 500
N_x = N_y = 60
x_min, x_max = -5, 5
y_min, y_max = -5, 5
x = np.linspace(x_min, x_max, N_x)
y = np.linspace(y_min, y_max, N_y)
t = np.linspace(0, 10, N_t)
dt = t[1] - t[0]
dx = x[1] - x[0]
dy = y[1] - y[0]
nu = 0.3


def condicion_inicial(x, y):
    return 5 * np.exp(-(x**2 + y**2))

u = np.zeros((N_t, N_x, N_y))
u[0, :, :] = condicion_inicial(np.repeat(x[:, np.newaxis], N_y, axis=1),
                                np.repeat(y[np.newaxis, :], N_x, axis=0))


for k in range(1, N_t):
    for i in range(1, N_x - 1):
        for j in range(1, N_y - 1):
            u[k, i, j] = u[k-1, i, j] - dt * (u[k-1, i, j] * (u[k-1, i+1, j] - u[k-1, i-1, j]) / (2*dx) +
                                              u[k-1, i, j] * (u[k-1, i, j+1] - u[k-1, i, j-1]) / (2*dy)) +\
                                        nu * dt * ((u[k-1, i+1, j] - 2*u[k-1, i, j] + u[k-1, i-1, j]) / dx**2 +
                                                   (u[k-1, i, j+1] - 2*u[k-1, i, j] + u[k-1, i, j-1]) / dy**2)

    u[k, 0, :] = 0
    u[k, -1, :] = 0
    u[k, :, 0] = 0
    u[k, :, -1] = 0


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)

# ChatGPT nos ayudó a esta parte de actualizar el gráfico porque la verdad aún nos cuesta mucho 
# el tema de graficar 3D.

def actualizar_grafico(numero_frame, z, grafico):
    grafico[0].remove()
    grafico[0] = ax.plot_surface(X, Y, z[numero_frame], cmap="viridis", edgecolor='none')

grafico = [ax.plot_surface(X, Y, u[0], cmap="viridis", edgecolor='none')]
ax.set_title('CangreBurgers jaja')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(0, 5)

ani = animation.FuncAnimation(fig, actualizar_grafico, frames=N_t, fargs=(u, grafico), interval=50, blit=False)

plt.show()
