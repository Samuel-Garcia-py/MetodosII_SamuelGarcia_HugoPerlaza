#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt

G = 39.5  # Constante gravitacional en unidades astronómicas^3 / año^2 / masa solar
M_solar = 1  # Masa del sol en unidades de masa solar
au = 1.496e11  # Unidad astronómica en metros
alpha = 1.1e-8  # Constante de corrección en au^2

# Parámetros de la órbita de Mercurio
e = 0.205630
a = 0.387098 * au

# Condiciones iniciales en el afelio
r0 = np.array([a * (1 + e), 0.])
v0 = np.array([0., np.sqrt((G * M_solar * (1 - e)) / (a * (1 + e)))])

# Paso temporal
dt = alpha  # Paso temporal del mismo orden de alpha

# Función de aceleración
def aceleracion(r):
    r_mag = np.linalg.norm(r)
    return -G * M_solar * r / (r_mag**3) * (1 + alpha / r_mag**2)

# Método de Verlet modificado
def verlet_modificado(r0, v0, dt, num_orbitas):
    orbit_period = 88  # Período orbital de Mercurio en días
    num_steps = int((orbit_period / 365.25) / dt) * num_orbitas  # Convertir el tiempo de días a años
    r = np.zeros((num_steps, 2))
    v = np.zeros((num_steps, 2))
    r[0] = r0
    v[0] = v0

    for i in range(num_steps - 1):
        r[i + 1] = r[i] + v[i] * dt + 0.5 * aceleracion(r[i]) * dt**2
        v[i + 1] = v[i] + 0.5 * (aceleracion(r[i + 1]) + aceleracion(r[i])) * dt

    return r

# Simulación para 10 órbitas
num_orbitas = 10
r = verlet_modificado(r0, v0, dt, num_orbitas)

# Graficar la trayectoria de Mercurio
plt.figure(figsize=(8, 8))
for i in range(num_orbitas):
    plt.plot(r[i::num_orbitas, 0], r[i::num_orbitas, 1], label=f'Órbita {i + 1}', alpha=0.5)
plt.plot(0, 0, 'o', label='Sol', color='Blue')  # Representar el Sol
plt.xlabel('x (UA)')
plt.ylabel('y (UA)')
plt.title('Órbitas de Mercurio')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()


# In[ ]:




