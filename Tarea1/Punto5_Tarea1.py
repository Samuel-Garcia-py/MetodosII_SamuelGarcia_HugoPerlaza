# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 22:44:41 2024

@author: Samuel García
"""


import numpy as np
import matplotlib.pyplot as plt


def f(t, u, alpha):
    return alpha * u


def runge_kutta(alpha, u0, dt, t_final):
    num_steps = int(t_final / dt)
    t_values = np.linspace(0, t_final, num_steps + 1)
    u_values = np.zeros(num_steps + 1)
    u_values[0] = u0

    for i in range(num_steps):
        k1 = f(t_values[i], u_values[i], alpha)
        k2 = f(t_values[i] + 0.5*dt, u_values[i] + 0.5*dt*k1, alpha)
        k3 = f(t_values[i] + 0.5*dt, u_values[i] + 0.5*dt*k2, alpha)
        k4 = f(t_values[i] + dt, u_values[i] + dt*k3, alpha)
        u_values[i+1] = u_values[i] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return t_values, u_values


alpha = -1  
u0 = 1      
t_final = 5 
dt_values = [1.1, 1.5, 1.9]  


def exact_solution(t, alpha):
    return np.exp(alpha * t)


plt.figure(figsize=(10, 6))

for dt in dt_values:
    t_values, u_values = runge_kutta(alpha, u0, dt, t_final)
    plt.plot(t_values, u_values, label=f'Δt = {dt}')

t_exact = np.linspace(0, t_final, 1000)
plt.plot(t_exact, exact_solution(t_exact, alpha), 'k--', label='Solución Exacta')

plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('Solución Numérica')
plt.legend()
plt.grid(True)
plt.show()

