# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:11:10 2024

@author: samue
"""


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
        
        # Historial
        
        self.R = np.zeros((len(t),len(r0)))
        self.V = np.zeros_like(self.R)
        self.A = np.zeros_like(self.R)
        
        self.F = np.zeros_like(self.R)
        
        self.P = np.zeros_like(self.R)
    
        # Fisica 
        self.K = 20.     
        
        self.VEk = np.zeros(len(t))
        
    def Evolution(self,i):
        
        
        self.SetPosition(i)
        self.SetVelocity(i)
        
        self.a = self.f/self.m
        
        # Euler
       # self.r += self.dt*self.v
       # self.v += self.dt*self.a
        
        # Euler-Cromer
        self.v += self.dt*self.a
        self.r += self.dt*self.v
        
        
    def CalculateForce(self,p): 
        
        d = np.linalg.norm(self.r - p.r)
        
        compresion = self.radius + p.radius - d
        
        if compresion >= 0:
            
            Fn = self.K * compresion**3
            
            self.n = (self.r - p.r)/d
            
            
            self.f = Fn*self.n
            
    def ResetForce(self):
        self.f[:] = 0.
        self.a[:] = 0.
        
    def CalculatePotentialEnergy(self, p2):
        d = np.linalg.norm(self.r - p2.r)
        compression = self.radius + p2.radius - d
        if compression >= 0:
            U = 0.5 * self.K * compression**2
            return U  
        else:
            return 0  
        
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
    
    def WallInteraction(self, box_size):
        
        self.r = np.clip(self.r, -box_size / 2. + self.radius, box_size / 2. - self.radius)
    
        # Velocidades inversas
        self.v = np.where(self.r - self.radius <= -box_size / 2., abs(self.v), self.v)
        self.v = np.where(self.r + self.radius >= box_size / 2., -abs(self.v), self.v)
                
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
        x0=np.random.uniform(-20,20)
        y0=np.random.uniform(-20,20)
        r=np.array([x0,y0])
        v0=np.array([np.random.uniform(-5,5),np.random.uniform(-5,5)])
        a0=np.array([0,0])
        p0 = Particle(r,v0,a0,t,m=1,radius=2,Id=i)
        Particles.append(p0)
    
    # Aca deber agregar una rutina montecarlo para crear particulas
    # sobre el plano con velocidades aleatorias.
    
    return Particles 


dt = 0.001

dt = 0.001
tmax = 10.0
t = np.arange(0,tmax,dt)
Particles = GetParticles(10,t)

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
scale = 200
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
        
        circle = plt.Circle( (x,y), p.radius, color='r', fill=False )
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

for i in range(1,len(Particles)):
    MomentumT = np.add(MomentumT,Particles[i].GetMomentum(scale))
    
fig3 = plt.figure(figsize=(10,5))
ax3 = fig3.add_subplot(221)
ax3.plot(t1,MomentumT[:,0],label='px')
ax3.plot(t1,MomentumT[:,1],label='py')
ax3.legend()

# a) y b)
fig4 = plt.figure(figsize=(10, 5))
ax4 = fig4.add_subplot(111)  
ax4.plot(t, total_px, label='Total px')
ax4.plot(t, total_py, label='Total py')
ax4.set_xlabel('Time')
ax4.set_ylabel('Total Linear Momentum')
ax4.legend()
plt.show()

# d) Energía cinética
fig_kinetic = plt.figure(figsize=(10, 5))
ax_kinetic = fig_kinetic.add_subplot(111)
ax_kinetic.plot(t, np.sum([p.GetKineticEnergy() for p in Particles], axis=0))
ax_kinetic.set_xlabel('Time')
ax_kinetic.set_ylabel('Total Kinetic Energy')
plt.show()

# e) Energía Potencial
total_potential_energy = np.zeros(len(t))
for i in range(len(Particles)):
    for j in range(i+1, len(Particles)):
        for it in range(len(t)):
            total_potential_energy[it] += Particles[i].CalculatePotentialEnergy(Particles[j])
fig_potential = plt.figure(figsize=(10, 5))
ax_potential = fig_potential.add_subplot(111)
ax_potential.plot(t, total_potential_energy)
ax_potential.set_xlabel('Time')
ax_potential.set_ylabel('Total Potential Energy')
plt.show()

# f) Energía potencial
total_kinetic_energy = np.sum([p.GetKineticEnergy() for p in Particles], axis=0)
fig_mechanical = plt.figure(figsize=(10, 5))
ax_mechanical = fig_mechanical.add_subplot(111)
ax_mechanical.plot(t, total_kinetic_energy + total_potential_energy)
ax_mechanical.set_xlabel('Time')
ax_mechanical.set_ylabel('Total Mechanical Energy')
plt.show()

# h) Momento angular
angular_momentum_z = np.zeros(len(t))
for i in range(len(t)):
    angular_momentum_z[i] = sum([p.CalculateAngularMomentumZ() for p in Particles])

fig_angular = plt.figure(figsize=(10, 5))
ax_angular = fig_angular.add_subplot(111)
ax_angular.plot(t, angular_momentum_z)
ax_angular.set_xlabel('Time')
ax_angular.set_ylabel('Angular Momentum (Lz)')
plt.show()


