# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:58:38 2024

@author: Samuel García & Hugo Perlaza
"""

#                                            Falling Ball

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm

class Particle:
    
    def __init__(self, r0, v0, a0, t, m=1, radius=2., Id=0):
        
        self.dt = t[1] - t[0]
        
        
        # Atributos instantaneos
        self.r = r0
        self.v = v0
        self.a = a0
        
        self.m = m
        self.radius = radius
        self.Id = Id
        
        
        self.p = self.m*self.v
        
        self.f = self.m*self.a
        
        self.gravity = np.array([0, -9.81])
        
        # Historial
        
        self.R = np.zeros((len(t),len(r0)))
        self.V = np.zeros_like(self.R)
        self.A = np.zeros_like(self.R)
        
        self.F = np.zeros_like(self.R)
        
        self.P = np.zeros_like(self.R)
    
        # Fisica 
        self.K = 20.     
        
        self.sigma = 1.5*self.radius
        
        self.epsilon = 1500
        
        self.VEk = np.zeros(len(t))
        self.Ep = 0.
        self.VEp = np.zeros(len(t))
        
    def Evolution(self,i):
        
        
        self.SetPosition(i)
        self.SetVelocity(i)
        
        self.a = self.f / self.m + self.gravity
        
        # Euler
       # self.r += self.dt*self.v
       # self.v += self.dt*self.a
        
        # Euler-Cromer
        self.v += self.dt*self.a
        self.v += self.dt * self.a
        
        
    def CalculateForce(self,p):
        
        d = np.linalg.norm(self.r - p.r)
        
        Fn = 4*self.epsilon*( 12*self.sigma**12/d**13 - 6*self.sigma**6/d**7  )
        
        self.n = (self.r - p.r)/d     
        
        self.f = np.add(self.f,Fn*self.n)
        
        # Falta implementar energía potencial 
        self.Ep += 4*self.epsilon*( self.sigma**12/d**12 - self.sigma**6/d**6  )
        
    def CalculatePotentialEnergy(self, p2):
        d = np.linalg.norm(self.r - p2.r)
        compression = self.radius + p2.radius - d
        if compression >= 0:
            U = 0.5 * self.K * compression**2
            return U  
        else:
            return 0 
            
    def ResetForce(self):
        self.f[:] = 0.
        self.a[:] = 0.
        self.Ep = 0.
        
        
    # Setter
    def SetPosition(self,i):
        self.R[i] = self.r
    
    def SetVelocity(self,i):
        self.V[i] = self.v
        self.P[i] = self.m*self.v
        self.VEk[i] = 0.5*self.m*np.dot(self.v,self.v)
    
    # Getter
    def GetPosition(self,scale=1):
        return self.R[::scale]
    
    def GetVelocity(self,scale=1):
        return self.V[::scale]
 
    def GetMomentum(self,scale=1):
        return self.P[::scale]
    
    def GetKineticEnergy(self,scale=1):
        return self.VEk[::scale] 
    
    def GetPotentialEnergy(self,scale=1):
        return self.VEp[::scale] 
    
    def WallInteraction(self, box_size, e=0.9):
        
        self.r = np.clip(self.r, -box_size / 2. + self.radius, box_size / 2. - self.radius)
    
        # Aplicar pérdida de energía en las colisiones con las paredes
        self.v = np.where(self.r - self.radius <= -box_size / 2., -e * self.v, self.v)
        self.v = np.where(self.r + self.radius >= box_size / 2., -e * self.v, self.v)

                
    def CalculateAngularMomentumZ(self):
       
        
        # Momento de inercia
        I = self.m * np.dot(self.r, self.r)
        
        # Velocidad angular
        omega_z = np.cross(self.v, np.array([0, 0])) / np.linalg.norm(self.r)
        
        # Momento angular: Yo considerio a priori que debería ser 0 pues las particulas no rotan
        Lz = I * omega_z
        
        return Lz
    
def GetParticles(N,t):
    
    Particles=[]
    for i in range(N):
        x0=np.random.uniform(-15,10)
        y0=np.random.uniform(-15,10)
        r=np.array([x0,y0])
        v0=np.array([np.random.uniform(0,2),np.random.uniform(0,2)])
        a0=np.array([0,0])
        p0 = Particle(r,v0,a0,t,m=1,radius=2,Id=i)
        Particles.append(p0)
    
    # Aca deber agregar una rutina montecarlo para crear particulas
    # sobre el plano con velocidades aleatorias.
    
    return Particles 

dt = 0.001
tmax = 30
t = np.arange(0,tmax,dt)
Particles = GetParticles(1,t)

def RunSimulation(t, Particles, box_size=40.):
    total_potential_energy = np.zeros(len(t))  
    total_px = np.zeros(len(t))  
    total_py = np.zeros(len(t))  
    
    for it in tqdm(range(len(t)), desc='Running simulation', unit=' Steps'):
        px = 0 
        py = 0 
        for i in range(len(Particles)):
            for j in range(len(Particles)):
                    
                if i != j:
                    Particles[i].CalculateForce(Particles[j])
                    
                if i != j: 
                    total_potential_energy[it] += Particles[i].CalculatePotentialEnergy(Particles[j])
                    
            Particles[i].v += dt * Particles[i].a
            Particles[i].r += dt * Particles[i].v

            Particles[i].WallInteraction(box_size)  
            Particles[i].Evolution(it)
            Particles[i].ResetForce()
            
            px += Particles[i].p[0]
            py += Particles[i].p[1] 
            
        total_px[it] = px
        total_py[it] = py
        

    return Particles, total_potential_energy, total_px, total_py # New

Particles, total_potential_energy, total_px, total_py = RunSimulation(t,Particles)

# Bajamos dimensión de la simulacion
scale = 100
t1 = t[::scale]

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)

def init():
    
    ax.clear()
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    
def Update(i):
    
    init()
    ax.set_title(r't =  %.3f s' %(t1[i]))
    
    
    # Queremos calcular la energía total de cinética
    KE = 0. # Kinetic energy
    
    for p in Particles:
        
        x = p.GetPosition(scale)[i,0]
        y = p.GetPosition(scale)[i,1]
        
        vx = p.GetVelocity(scale)[i,0]
        vy = p.GetVelocity(scale)[i,1]
        
        circle = plt.Circle( (x,y), p.radius, color='r', fill=True )
        ax.add_patch(circle)
        
        ax.arrow(x,y,vx,vy,color='k',head_width=0.5,length_includes_head=True)
        
        KE += p.GetKineticEnergy(scale)[i]
        
        ax1.set_title(r'Total kinetic Energy: {:.3f}'.format(KE))
        ax1.scatter(t1[:i], p.GetKineticEnergy(scale)[:i],color='k',marker='.')
        
Animation = anim.FuncAnimation(fig,Update,frames=len(t1),init_func=init)

Writer = anim.writers['ffmpeg']
writer_ = Writer(fps=10, metadata=dict(artist='Me'))
#Animation.save('EsferaDura.mp4', writer=writer_)

MomentumT = Particles[0].GetMomentum(scale)
EnergyT = Particles[0].GetKineticEnergy(scale)
EnergyP = Particles[0].GetPotentialEnergy(scale)
EnergyP *= 0.5

for i in range(1,len(Particles)):
    MomentumT = np.add(MomentumT,Particles[i].GetMomentum(scale))
    EnergyT = np.add(EnergyT,Particles[i].GetKineticEnergy(scale))
    EnergyP = np.add(EnergyP,Particles[i].GetPotentialEnergy(scale))
    

'''

Esto es de la tarea, es que reciclé el código
# d) Energía cinética
fig_kinetic = plt.figure(figsize=(10, 10))


# e) Energía Potencial
total_potential_energy = np.zeros(len(t))
for i in range(len(Particles)):
    for j in range(i+1, len(Particles)):
        for it in range(len(t)):
            total_potential_energy[it] += Particles[i].CalculatePotentialEnergy(Particles[j])


# f) Energía mecánica
total_kinetic_energy = np.sum([p.GetKineticEnergy() for p in Particles], axis=0)


# h) Momento angular
angular_momentum_z = np.zeros(len(t))
for i in range(len(t)):
    angular_momentum_z[i] = sum([p.CalculateAngularMomentumZ() for p in Particles])
    
'''

# b) Cuánto tarda antes de de dejar de rebotar: 23 segundos

# c)

altura_maxima = float('-inf')  

alturas_iniciales = []

for i in range(len(t)):
    altura_particula = np.max(Particles[0].GetPosition()[i])
    
    if altura_particula > altura_maxima:
        altura_maxima = altura_particula

    if i > 0 and Particles[0].GetVelocity()[i, 1] * Particles[0].GetVelocity()[i - 1, 1] < 0:
        alturas_iniciales.append(altura_particula)


print("Altura máxima: :", altura_maxima)

restituciones = []

for i in range(1, len(alturas_iniciales)):
    h0 = alturas_iniciales[i - 1]
    h1 = altura_maxima
    e = np.sqrt(h1 / h0)
    restituciones.append(e)

print("restitución:", restituciones)







