import numpy as np
from APFObstacle import APFObstacle
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math
#import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.cm as cm

obstacles = np.array([[40,50]])
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

k=0.04
i=10000000000000000000
repAttRatio = 1.3

#to do:
# 1) add robot symbol
# 2) create animation loop
# 3) move robot symbol according to force data
# 4) actually get video library running
# 5) create video

#other
#figure out why i component of sigmoid repulsion limit isn't working.
#mayble implement piecewise turnoff for repulsion
#

def calcFatt(xi,pG,KV,VG,V):
    #Fatt = np.dot(-xi,pG) + (KV*(VG-V))
    u = -xi * pG[0] + (KV * (VG-V))[0] 
    v = -xi * pG[1] + (KV * (VG-V))[1]
    Fatt = [u,v]
    Fatt = np.multiply((1/repAttRatio),Fatt/magnitude([u,v]))
    #print("Fatt: " + str(Fatt))
    return Fatt


def calcpG(goalPos,currentPos):
    #print(np.shape(goalPos))
    #print(np.shape(currentPos))
    u = currentPos[0] - goalPos[0]
    v = currentPos[1] - goalPos[1]
    pG = [u,v]
    #print("pG: " + str(pG))
    return pG

def magnitude(vector):
    return np.sqrt(sum(pow(element, 2) for element in vector))

def calcFrep(x,y,obsPos):
    #print(np.shape(obsPos))
    #print(obsPos)
    #print(currentPos)
    #pQ = calcpG(obsPos, currentPos)
    #print(pQ)
    #Frep = self.nu*(pQ**-1 - self.p0**-1)*((pQ**-1)**2)*np.power((currentPos-self.goalPos),self.n)
    
    #building my own repulsion because this one sucks
    #print(obsPos[0])
    #print(obsPos[1])
    u = -((y-obsPos[1])-(x-obsPos[0]))#/np.sqrt(sum(pow(-(y-x),2), pow(-(-y-x),2)))
    v = -(-(y-obsPos[1])-(x-obsPos[0]))#/np.sqrt(sum(pow(-(y-x),2), pow(-(-y-x),2)))
    #print(u)
    
    #u = nu*((1/pQ[0])-(1/p0)*((1/pQ[0])**2))#*(currentPos[0]-obsPos[0]))
    #v = nu*((1/pQ[1])-(1/p0)*((1/pQ[1])**2))#*(currentPos[1]-obsPos[1]))
    
    #u = nu*(np.power(pQ[0],-1) - np.power(float(p0),-1))*(np.power(float(pQ[0]),-1)**2)#*np.power((currentPos[0] - goalPos[0]),n))
    #v = nu*(pQ[1]**-1 - p0**-1)*((pQ[1]**-1**2))#*np.power((currentPos[1] - goalPos[1]),n))
    Frep = [u,v]#/(magnitude([u,v])))#*(1/(1 + np.exp(k*(magnitude([u,v])-i))))
    #print(Frep)
    #print("before Scaleup")
    #print(Frep)
    #Frep = np.multiply(Frep, 200)
    #Frep = Frep * 200
    Frep = (Frep/(magnitude([u,v])))*1/(1+np.exp(k*(magnitude([u,v])-i)))*(repAttRatio/1)
    #Frep = (100*Frep).astype(int)
    #print("after Scaleup")
    #print(Frep)
    #Frep = Frep * np.full(np.shape(Frep), 1)
    #print("after Scaleup 2")
    #print(Frep)
    return Frep

uatt = []
vatt = []

x, y = np.meshgrid(np.arange(0, 80, 5, dtype=float),
                   np.arange(0, 80, 5, dtype=float))

uatt = calcFatt(xi,calcpG(goalPos,[x,y]),KV,VG,V)[0]
vatt = calcFatt(xi,calcpG(goalPos,[x,y]),KV,VG,V)[1]
Fatt = [uatt, vatt]#np.multiply([uatt, vatt],(1/repAttRatio))



F = Fatt #np.zeros((10,10))#Fatt

for i,obsPos in enumerate(obstacles):
    urep = calcFrep(x,y,obsPos)[0]
    vrep = calcFrep(x,y,obsPos)[1]
    #print(F)
    Frep = [urep, vrep]
    F = np.add(F, Frep)
#print(type(F))
#print(np.shape(F))
#print(F)
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

robotPos = [0,40]
frames = []
fig, ax = plt.subplots()
for i in np.arange(0,200,1):
    uRoatt = calcFatt(xi,calcpG(goalPos,[robotPos[0],robotPos[1]]),KV,VG,V)[0]
    vRoatt = calcFatt(xi,calcpG(goalPos,[robotPos[0],robotPos[1]]),KV,VG,V)[1]
    #print("vRoatt: " + str(vRoatt))
    FRoatt = [uRoatt, vRoatt]
    #FRoatt = np.multiply([uRoatt, vRoatt],(1/repAttRatio))
    #print("FRoatt: " + str(FRoatt))
    
    FRo = FRoatt
    #print(FRo)
    
    #Frep = [0,0]
    for j,obsPos in enumerate(obstacles):
        uRorep = calcFrep(robotPos[0],robotPos[1], obsPos)[0]
        vRorep = calcFrep(robotPos[0],robotPos[1], obsPos)[1]
        #print(uRorep)
        FRorep = [uRorep, vRorep]
        #print(Frep)
        FRo = np.add(FRo, FRorep)
    
    print("iteration: " + str(i))
    print("robotPos: " + str(robotPos))
    print("FRo: " + str(FRo))
    robotPos = robotPos + FRo
    plt.quiver(robotPos[0], robotPos[1], FRoatt[0], FRoatt[1])

    plt.quiver(x, y, F[0], F[1])
    plt.plot(goalPos[0],goalPos[1],'x')
    plt.plot(robotPos[0],robotPos[1],'o', color="red")
    plt.grid()
    plt.savefig("Images/PotentialField_" + str(i))
    #generate_video(plt.show())
    #frames.append([plt.show(animated=True)])
    #im = plt.imshow(fig)
    #r = plt.gcf().canvas.get_renderer()
    #x = im.make_image(r, magnification=1.0)
    #frames.append(x)
    #im = plt.imshow(plt.gcf().canvas.get_renderer())
    #frames.append([im])

#ani = animation.FuncAnimation(fig, animate, interval = 1, blit=True, repeat_delay=1000)
#ani.save("ArtificialPotentialFieldNavigation.mp4")
#plt.show()
#print(FattList)
