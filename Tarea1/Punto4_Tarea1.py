#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm 


# In[16]:


class Planeta:
    
    def __init__(self, e, a, t):
        
        self.t = t
        self.dt = t[1] - t[0] # Paso del tiempo
        
        self.e = e # Excentricidad
        self.a_ = a # Semi-eje mayor
        
        self.G = 4*np.pi**2 # Unidades gaussianas
        
        self.r = np.zeros(3)
        self.v = np.zeros_like(self.r)
        self.a = np.zeros_like(self.r)
        
        self.r[0] = self.a_*(1-self.e)
        self.v[1] = np.sqrt( self.G*(1+self.e)/(self.a_*(1.-self.e)) )
        
        self.R = np.zeros((len(t),len(self.r)))
        self.V = np.zeros_like(self.R)
        
        # El valor del pasado
        self.rp = self.r
        
    def GetAceleration(self):
        
        d = np.linalg.norm(self.r)
        self.a = -self.G/d**3*self.r
        
        
    def Evolution(self,i):
        
        self.SetPosition(i)
        self.SetVelocity(i)
        self.GetAceleration()
        
        if i==0:
            self.r = self.rp + self.v*self.dt
        else:
            
            # rp pasado, r presente rf futuro
            self.rf = 2*self.r - self.rp + self.a*self.dt**2
            self.v = (self.rf - self.rp)/(2*self.dt)
            
            self.rp = self.r
            self.r = self.rf
    
    def SetPosition(self,i):
        self.R[i] = self.r
        
    def SetVelocity(self,i):
        self.V[i] = self.v
    
    def GetPosition(self,scale=1):
        return self.R[::scale]
    
    def GetVelocity(self,scale=1):
        return self.V[::scale]
    
    def GetPerihelio(self):
        
        Dist = np.linalg.norm(self.R,axis=1)
        
        timeup = []
        
        for i in range(1,len(Dist)-1):
            if Dist[i] < Dist[i-1] and Dist[i] < Dist[i+1]:
                timeup.append(self.t[i])
            
        return timeup


# In[17]:


def GetPlanetas(t):
    
    Mercurio = Planeta(0.2056,0.307,t)
    Venus = Planeta(0.0067,0.7233,t)
    Tierra = Planeta(0.01671,1.,t)
    Marte = Planeta(0.0934,1.524,t)
    Jupiter = Planeta(0.0483,5.20440,t)
    
    return [Mercurio,Venus,Tierra,Marte,Jupiter]


# In[18]:


dt = 0.001
tmax = 20
t = np.arange(0.,tmax,dt)
Planetas = GetPlanetas(t)


# In[19]:


def RunSimulation(t,Planetas):
    
    for it in tqdm(range(len(t)), desc='Running simulation', unit=' Steps' ):
        
        #print(it)
        for i in range(len(Planetas)):
            Planetas[i].Evolution(it)
            # Aca debes agregar la interaccion con la pared
            
            
    return Planetas


# In[20]:


Planetas = RunSimulation(t,Planetas)


# In[21]:


Planetas[1].GetPerihelio()


# In[22]:


scale = 20
t1 = t[::scale]


# In[23]:


#plt.plot(Planetas[0].GetPosition()[:,0],Planetas[0].GetPosition()[:,1])

periodos_orbitales = []

# Iterar sobre cada planeta para calcular los periodos orbitales
for planeta in Planetas:
    perihelios = planeta.GetPerihelio()  # Obtener los tiempos de perihelio del planeta
    periodos = []  # Lista para almacenar los periodos orbitales del planeta
    
    # Calcular la diferencia de tiempo entre cada par de perihelios consecutivos
    for i in range(len(perihelios) - 1):
        periodo = perihelios[i + 1] - perihelios[i]
        periodos.append(periodo)
    
    # Calcular el promedio de los periodos si la lista no está vacía
    if len(periodos) > 0:
        periodo_promedio = sum(periodos) / len(periodos)
        # Agregar el periodo orbital a la lista de periodos orbitales
        periodos_orbitales.append(periodo_promedio)
    else:
        # Si no hay perihelios detectados, podemos asignar un valor predeterminado o None
        periodos_orbitales.append(None)
        



# In[24]:


print(periodos_orbitales)


# In[25]:


# Crear listas para almacenar los semi-ejes mayores y los periodos orbitales
semi_ejes_cubicos = []
periodos_cuadrados = []

# Iterar sobre cada planeta para calcular los semi-ejes mayores y los periodos orbitales
for planeta, periodo_orbital in zip(Planetas, periodos_orbitales):
    # Verificar si el periodo orbital es None (no se detectaron suficientes perihelios)
    if periodo_orbital is not None:
        semi_eje_cubico = planeta.a_ ** 3
        periodo_cuadrado = periodo_orbital ** 2
        semi_ejes_cubicos.append(semi_eje_cubico)
        periodos_cuadrados.append(periodo_cuadrado)

# Graficar T^2 vs. a^3
plt.figure(figsize=(8, 6))
plt.scatter(semi_ejes_cubicos, periodos_cuadrados, color='blue')

# Etiquetas y título del gráfico
plt.xlabel('Semi-eje mayor al cubo (a^3)')
plt.ylabel('Periodo al cuadrado (T^2)')
plt.title('Periodo al Cuadrado vs. Semi-eje Mayor al Cubo')

# Mostrar la cuadrícula
plt.grid(True)

# Mostrar el gráfico
plt.show()


# In[26]:


# Convertir las listas de datos a numpy arrays para realizar la regresión lineal
x = np.array(semi_ejes_cubicos)
y = np.array(periodos_cuadrados)

# Realizar la regresión lineal usando numpy
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]  # Calcular la pendiente (m) y el punto de corte (c)

# Graficar los datos junto con la línea de regresión
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Datos')  # Datos originales
plt.plot(x, m*x + c, color='red', label=f'Regresión lineal: y = {m:.2f}x + {c:.2f}')  # Línea de regresión
plt.xlabel('Semi-eje mayor al cubo (a^3)')
plt.ylabel('Periodo al cuadrado (T^2)')
plt.title('Regresión Lineal: Periodo al Cuadrado vs. Semi-eje Mayor al Cubo')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir la pendiente y el punto de corte
print(f"Pendiente (m): {m}")
print(f"Punto de corte (c): {c}")


# In[31]:


# Definir la constante gravitacional en el SI
G_SI = 6.674 * 10**-11  # m^3 kg^-1 s^-2

segundos_por_año = 365.25 * 24 * 3600
pendiente = m * segundos_por_año**2

# Conversión de AU al cubo a metro cúbico
# 1 AU = 149597870.7 km, 1 km = 1000 metros
metros_por_AU = 149597870.7 * 1000
volumen_metro_cubico = (metros_por_AU)**3

pendiente_s2_m3 = pendiente / volumen_metro_cubico

# Imprimir el resultado de la conversión
print("Pendiente en segundos cuadrado por metro cúbico:", pendiente_s2_m3, "s^2/m^3")

# Resolver para la masa del Sol en el SI
masa_sol_SI = (4*np.pi**2) / (G_SI * pendiente_s2_m3)

# Convertir la masa del Sol al SI a unidades gaussianas
masa_sol_gaussian = masa_sol_SI * 10**3  # Convertir de kg a g

# Imprimir los resultados
print("Masa del Sol en el SI:", masa_sol_SI, "kg")
print("Masa del Sol en unidades gaussianas:", masa_sol_gaussian, "g")


# In[14]:


fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(221,projection='3d')
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)

colors=['r','k','b']

def init():
    
    ax.clear()
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    
    ax1.clear()
    ax1.set_xlim(-1,1)
    ax1.set_ylim(-1,1) 
    
    ax2.clear()
    ax2.set_xlim(-2,2)
    ax2.set_ylim(-2,2) 
    
def Update(i):
    
    init()
    
    for j, p in enumerate(Planetas):
        
        x = p.GetPosition(scale)[i,0]
        y = p.GetPosition(scale)[i,1]
        z = p.GetPosition(scale)[i,2]
        
        vx = p.GetVelocity(scale)[i,0]
        vy = p.GetVelocity(scale)[i,1]
        vz = p.GetVelocity(scale)[i,2]
    
        ax.scatter(0,0,0,s=200,color='y')
        ax.quiver(x,y,z,vx,vy,vz,color=colors[j],length=0.03)
        
        ax.scatter(x,y,z,color=colors[j])
        
        circle = plt.Circle((x,y),0.1,color=colors[j],fill=True)
        ax1.add_patch(circle)
    
    # Mercurio visto desde tierra
    Mx = Planetas[0].GetPosition(scale)[:i,0] - Planetas[2].GetPosition(scale)[:i,0]
    My = Planetas[0].GetPosition(scale)[:i,1] - Planetas[2].GetPosition(scale)[:i,1]
    
    # Venus visto desde tierra
    Vx = Planetas[1].GetPosition(scale)[:i,0] - Planetas[2].GetPosition(scale)[:i,0]
    Vy = Planetas[1].GetPosition(scale)[:i,1] - Planetas[2].GetPosition(scale)[:i,1]
    
    ax2.scatter(Mx,My,marker='.',label='Mercurio')
    ax2.scatter(Vx,Vy,marker='.',label='Venus')
    
Animation = anim.FuncAnimation(fig,Update,frames=len(t1),init_func=init)


# In[ ]:




