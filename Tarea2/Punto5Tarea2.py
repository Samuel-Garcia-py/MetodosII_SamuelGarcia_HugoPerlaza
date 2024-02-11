#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# Método de Runge-Kutta de segundo orden
def runge_kutta(f, u0, t):
    n = len(t)
    u = np.zeros(n)
    u[0] = u0
    dt = t[1] - t[0]
    
    for i in range(1, n):
        k1 = f(u[i-1], t[i-1])
        k2 = f(u[i-1] + dt*k1, t[i-1] + dt)
        
        u[i] = u[i-1] + 0.5 * dt * (k1 + k2)
        
    return u

# Función que define la ecuación diferencial
def ecuacion(u, q):
    return u**q

# Valores de q
q_valores = [0.0, 0.2, 0.4, 0.7, 0.9, 1.0]

# Condiciones iniciales y rango de tiempo
u0 = 1.0  # Condición inicial
t = np.linspace(0, 10, 1000)  # Rango de tiempo

# Graficar las soluciones
for q in q_valores:
    # Resolver la ecuación diferencial usando el método de Runge-Kutta de segundo orden
    sol = runge_kutta(lambda u, t: ecuacion(u, q), u0, t)
    
    # Graficar la solución
    plt.plot(t, sol, label=f'q={q}')

# Configuración de la gráfica
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('Solución de la ecuación diferencial')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




