# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:19:35 2024

@author: samue
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Definir las constantes del problema
G = 6.67e-11  # Constante de gravitación universal (m^3/kg/s^2)
m_t = 5.9736e24  # Masa de la Tierra (kg)
m_L = 0.07349e24  # Masa de la Luna (kg)
r_T = 6.3781e6  # Radio de la Tierra (m)
r_L = 1.7374e6  # Radio de la Luna (m)
d = 3.844e8  # Distancia de la Tierra a la Luna (m)
ω = 2.6617e-6  # Frecuencia angular de la Luna (1/s)

# Definir las ecuaciones de movimiento y las fuerzas
def Hamiltonian_equations(t, state):
    p_r, p_phi, r, phi = state
    r_L_value = np.sqrt(r**2 + d**2 - 2*r*d*np.cos(phi - ω*t))
    F_r = -partial_H_partial_r(p_r, p_phi, r, phi, t)
    F_phi = -partial_H_partial_phi(p_r, p_phi, r, phi, t)
    dp_r_dt = -F_r
    dp_phi_dt = -F_phi
    dr_dt = partial_H_partial_p_r(p_r, p_phi, r, phi)
    dphi_dt = partial_H_partial_p_phi(p_r, p_phi, r, phi)
    return [dp_r_dt, dp_phi_dt, dr_dt, dphi_dt]

def partial_H_partial_r(p_r, p_phi, r, phi,t):
    r_L_value = np.sqrt(r**2 + d**2 - 2*r*d*np.cos(phi - ω*t))
    return p_r / m_t + p_phi**2 / (m_t * r**3) - G * (m_t * m_L / r**2) - G * (m_t * m_L / r_L_value**2) * (r - d*np.cos(phi - ω*t)) / r_L_value

def partial_H_partial_phi(p_r, p_phi, r, phi, t):
    r_L_value = np.sqrt(r**2 + d**2 - 2*r*d*np.cos(phi - ω*t))
    return p_phi / (m_t * r**2) - G * (m_t * m_L / r_L_value**2) * (-d*r*np.sin(phi - ω*t)) / r_L_value

def partial_H_partial_p_r(p_r, p_phi, r, phi):
    return p_r / m_t

def partial_H_partial_p_phi(p_r, p_phi, r, phi):
    return p_phi / (m_t * r**2)

# Implementar el algoritmo de Runge-Kutta de cuarto orden
def runge_kutta_fourth_order(f, state0, t0, tf, h):
    t_values = np.arange(t0, tf + h, h)
    state = np.zeros((len(t_values), len(state0)))
    state[0] = state0

    for i in range(len(t_values) - 1):
        k1 = h * np.array(f(t_values[i], state[i]))
        k2 = h * np.array(f(t_values[i] + h/2, state[i] + k1/2))
        k3 = h * np.array(f(t_values[i] + h/2, state[i] + k2/2))
        k4 = h * np.array(f(t_values[i] + h, state[i] + k3))
        state[i+1] = state[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, state

# Configurar las condiciones iniciales y los parámetros
p_r0 = 0  # Momento lineal radial inicial
p_phi0 = 0  # Momento angular inicial
r0 = r_T  # Radio inicial (radio de la Tierra)
phi0 = np.pi / 2  # Latitud inicial (en el ecuador)
state0 = [p_r0, p_phi0, r0, phi0]  # Estado inicial
t0 = 0  # Tiempo inicial
tf = 86400 * 7  # Tiempo final (7 días terrestres)
h = 1  # Paso de integración (segundos de vuelo)

# Ejecutar la simulación
t_values, state_values = runge_kutta_fourth_order(Hamiltonian_equations, state0, t0, tf, h)

# Extraer las variables para facilitar la visualización
p_r_values, p_phi_values, r_values, phi_values = state_values[:,0], state_values[:,1], state_values[:,2], state_values[:,3]

# Crear la animación
fig, ax = plt.subplots()
ax.set_xlim(-d*1.5, d*1.5)
ax.set_ylim(-d*1.5, d*1.5)
ax.set_aspect('equal', adjustable='box')
ax.plot(0, 0, 'o', color='blue', markersize=10)  # Representación de la Tierra

line_nave, = ax.plot([], [], 'o', color='red', markersize=5)  # Representación de la nave
line_luna, = ax.plot([], [], 'o', color='gray', markersize=3)  # Representación de la Luna

def init():
    line_nave.set_data([], [])
    line_luna.set_data([], [])
    return (line_nave, line_luna)

def animate(i):
    x_nave = r_values[i] * np.cos(phi_values[i])
    y_nave = r_values[i] * np.sin(phi_values[i])
    x_luna = d * np.cos(ω * t_values[i])
    y_luna = d * np.sin(ω * t_values[i])
    
    line_nave.set_data(x_nave, y_nave)
    line_luna.set_data(x_luna, y_luna)
    
    return (line_nave, line_luna)

ani = FuncAnimation(fig, animate, init_func=init, frames=len(t_values), interval=100, blit=True)

plt.xlabel('Posición x (m)')
plt.ylabel('Posición y (m)')
plt.title('Simulación del movimiento de la nave espacial en el sistema Tierra-Luna')

plt.show()





