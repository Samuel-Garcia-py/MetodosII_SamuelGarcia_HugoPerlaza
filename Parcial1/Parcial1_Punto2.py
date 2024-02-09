#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parámetros físicos
g = 9.81  # Aceleración debido a la gravedad en m/s^2
L = 1.0   # Longitud de la barra en metros

# Definir la ecuación diferencial que describe el movimiento del centro de masas
def System(r, t):
    theta, omega = r
    dtheta_dt = omega
    domega_dt = - (3 * g / (2 * L)) * np.sin(theta)  # Ecuación diferencial para la aceleración angular
    return [dtheta_dt, domega_dt]

# Condiciones iniciales
theta_0 = np.radians(10)  # Convertir a radianes
omega_0 = 0.0

# Arreglo de tiempo
t = np.linspace(0, 10, 1000)  # De 0 a 10 segundos con 1000 puntos

# Resolver la ecuación diferencial
solution = odeint(System, [theta_0, omega_0], t)

# Extraer las soluciones x(t) e y(t) del centro de masas
x = (L / 2) * np.sin(solution[:, 0])  # x = (L / 2) * sin(theta)
y = (L / 2) * np.cos(solution[:, 0])  # y = (L / 2) * cos(theta)

# Graficar la trayectoria y vs x del centro de masas
plt.figure(figsize=(8, 6))
plt.plot(x[x >= 0], y[x >= 0], label='Trayectoria del centro de masas', color='blue')  # Cortar la gráfica en x >= 0
plt.title('Trayectoria del centro de masas')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid(True)
plt.legend()
plt.show()



# In[ ]:




