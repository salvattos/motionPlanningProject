import numpy as np
from APFObstacle import APFObstacle
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt



def calcFatt(xi,pG,KV,VG,V):
    #Fatt = np.dot(-xi,pG) + (KV*(VG-V))
    u = -xi * pG[0] + (KV * (VG-V))[0] 
    v = -xi * pG[1] + (KV * (VG-V))[1]
    Fatt = [u,v]
    return Fatt


def calcpG(goalPos,currentPos):
    #print(currentPos[0])
    #print(currentPos[1])
    #print(np.shape(currentPos))
    #print(goalPos)
    #print(np.shape(goalPos))
    #pG = np.subtract(currentPos,goalPos) #might need to switch order
    u = currentPos[0] - goalPos[0]
    v = currentPos[1] - goalPos[1]
    pG = [u,v]
    return pG


obstaclePos = np.array([50,50])
goalPos  = np.array([95,85])

startPos = np.array([10,15])
xi = .100
VG = np.array([0,0])
V0 = np.array([0,0])
V = np.array([0,0])
p0 = 5
KV = 60

OB1 = APFObstacle(obstaclePos,goalPos,p0,V0)

Frep = OB1.calcFrepTotal(obstaclePos+np.array([3,4]),V)
#FattList = []
uatt = []
vatt = []

x, y = np.meshgrid(np.arange(0, 100, 10),
                   np.arange(0, 100, 10))

#u,v = calcFatt(xi,calcpG(goalPos,np.array([x,y])),KV,VG,V)

u = []
v = []

#for x in np.arange(0,100,10):
#    for y in np.arange(0,100,10):
#        currentPos = np.array([x,y])
#        u.append(calcFatt(xi,calcpG(goalPos,currentPos),KV,VG,V)[0])
#        v.append(calcFatt(xi,calcpG(goalPos,currentPos),KV,VG,V)[1])
#        currentPos = np.array([x,y])
#        uatt.append(calcFatt(xi,calcpG(goalPos,currentPos),KV,VG,V)[0])
#        vatt.append(calcFatt(xi,calcpG(goalPos,currentPos),KV,VG,V)[1])
        #FattList.append(Fatt)
        #Frep = 0
        #Fsum = np.add(Frep, Fatt)
#u = np.array(u)
#v = np.array(v)
#u = 

u = calcFatt(xi,calcpG(goalPos,[x,y]),KV,VG,V)[0]
v = calcFatt(xi,calcpG(goalPos,[x,y]),KV,VG,V)[1]
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
plt.quiver(x, y, u, v)
plt.grid()
plt.show()

#print(FattList)
