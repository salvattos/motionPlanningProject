import numpy as np
from APFObstacle import APFObstacle
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math

obstacles = np.array([50,50])
goalPos  = np.array([60,60])

startPos = np.array([10,15])
xi = .100
VG = np.array([0,0])
V0 = np.array([0,0])
V = np.array([0,0])
p0 = 5.
KV = 60.

#goalPos = goalPos
#V0 = V0
#p0 = p0
#obstaclePos = obstaclePos   
nu = 80.
n = 2.
#KV = 60
KVPrime = 70.

k=0.03
i=100000000000000


def calcFatt(xi,pG,KV,VG,V):
    #Fatt = np.dot(-xi,pG) + (KV*(VG-V))
    u = -xi * pG[0] + (KV * (VG-V))[0] 
    v = -xi * pG[1] + (KV * (VG-V))[1]
    Fatt = [u,v]
    return Fatt


def calcpG(goalPos,currentPos):
    #print(np.shape(goalPos))
    #print(np.shape(currentPos))
    u = currentPos[0] - goalPos[0]
    v = currentPos[1] - goalPos[1]
    pG = [u,v]
    return pG

def magnitude(vector):
    return np.sqrt(sum(pow(element, 2) for element in vector))

def calcFrep(obsPos, currentPos):
    #print(np.shape(obsPos))
    #print(obsPos)
    pQ = calcpG(obsPos, currentPos)
    #Frep = self.nu*(pQ**-1 - self.p0**-1)*((pQ**-1)**2)*np.power((currentPos-self.goalPos),self.n)
    
    #building my own repulsion because this one sucks
    u = -(y-x)#/np.sqrt(sum(pow(-(y-x),2), pow(-(-y-x),2)))
    v = -(-y-x)#/np.sqrt(sum(pow(-(y-x),2), pow(-(-y-x),2)))
    
    #u = nu*((1/pQ[0])-(1/p0)*((1/pQ[0])**2))#*(currentPos[0]-obsPos[0]))
    #v = nu*((1/pQ[1])-(1/p0)*((1/pQ[1])**2))#*(currentPos[1]-obsPos[1]))
    
    #u = nu*(np.power(pQ[0],-1) - np.power(float(p0),-1))*(np.power(float(pQ[0]),-1)**2)#*np.power((currentPos[0] - goalPos[0]),n))
    #v = nu*(pQ[1]**-1 - p0**-1)*((pQ[1]**-1**2))#*np.power((currentPos[1] - goalPos[1]),n))
    Frep = [u,v]#/(magnitude([u,v])))#*(1/(1 + np.exp(k*(magnitude([u,v])-i))))
    print("before Scaleup")
    print(Frep)
    #Frep = np.multiply(Frep, 200)
    #Frep = Frep * 200
    Frep = Frep/(magnitude([u,v]))*1/(1+np.exp(k*(magnitude([u,v])-i)))
    #Frep = (100*Frep).astype(int)
    print("after Scaleup")
    print(Frep)
    #Frep = Frep * np.full(np.shape(Frep), 1)
    #print("after Scaleup 2")
    #print(Frep)
    return Frep

uatt = []
vatt = []

x, y = np.meshgrid(np.arange(0, 100, 10, dtype=float),
                   np.arange(0, 100, 10, dtype=float))

uatt = calcFatt(xi,calcpG(goalPos,[x,y]),KV,VG,V)[0]
vatt = calcFatt(xi,calcpG(goalPos,[x,y]),KV,VG,V)[1]
Fatt = [uatt, vatt]



F = np.zeros((10,10))#Fatt

for i,obsPos in enumerate(obstacles):
    urep = calcFrep([10,20], [x,y])[0]
    vrep = calcFrep([10,20], [x,y])[1]
    Frep = [urep, vrep]
    F = np.add(F, Frep)
print(type(F))
print(np.shape(F))
print(F)
#u = -(y-x)
#v = -(-y-x)
w = 0
#x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.3),
#                      np.arange(-0.8, 1, 0.3),
#                      np.arange(-0.8, 1, 0.3))

#u = -(y-x)#np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
#v = -(-y-x)#-np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
#w = -(-z-x)#(np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z))
#z=0
#print(z)

#print(x)
#print(y)
#for x in range(0,100):
#    for y in range(0,100):
#        currentPos = np.array([x,y])
        #Attraction force
#        Fatt = calcFatt(xi,calcpG(goalPos,currentPos),KV,VG,V)
#        FattList.append(Fatt)
        #Frep = OB1.calcFrepTotal(currentPos,V)
#        Frep = 0
#        Fsum = Frep+Fatt
#u, v = 
#fig = plt.figure()
#ax = fig.add_subplot(projection="3d")
plt.quiver(x, y, F[0], F[1])
plt.grid()
plt.show()

#print(FattList)
