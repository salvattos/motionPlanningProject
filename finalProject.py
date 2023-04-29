import numpy as np
from APFObstacle import APFObstacle
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm



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
    #Fatt = Fatt/magnitude([u,v])
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

def calcFrep(x,y,obsPos,k,repAttRatio):
    i = 1000000
    u = -((y-obsPos[1])-(x-obsPos[0]))#/np.sqrt(sum(pow(-(y-x),2), pow(-(-y-x),2)))
    v = -(-(y-obsPos[1])-(x-obsPos[0]))#/np.sqrt(sum(pow(-(y-x),2), pow(-(-y-x),2)))
    Frep = [u,v]#/(magnitude([u,v])))#*(1/(1 + np.exp(k*(magnitude([u,v])-i))))
    Frep = (Frep/(magnitude([u,v])))*1/(1+np.exp(k*(magnitude([u,v])-i)))*(repAttRatio/1)
    return Frep


uatt = []
vatt = []


w = 0


robotPos = np.array([0,0])
OB1 = APFObstacle([30,30],5,np.array([.1,.1]))
OB1.nu = 10
OB2 = APFObstacle([55,45],5,np.array([0,.1]))
OB3 = APFObstacle([65,85],5,np.array([0,-.1]))
OB4 = APFObstacle([85,75],5,np.array([0,0]))

obstacles = [OB1,OB2,OB3,OB4]

def animateSOC(frame):
    plt.grid()
    #plt.clf()
    global robotPos
    global obstacles
    goalPos  = np.array([90,90])


    xi = .02
    VG = np.array([0,0])
    V = np.array([0,0])
    KV = 60.

    repAttRatio = 1


    uRoatt = calcFatt(xi,calcpG(goalPos,[robotPos[0],robotPos[1]]),KV,VG,V)[0]
    vRoatt = calcFatt(xi,calcpG(goalPos,[robotPos[0],robotPos[1]]),KV,VG,V)[1]
    #print("vRoatt: " + str(vRoatt))
    FRoatt = np.multiply([uRoatt, vRoatt],(1/repAttRatio))
    
    FRo = calcFatt(xi,calcpG(goalPos,[robotPos[0],robotPos[1]]),KV,VG,V)

    for obs in obstacles:
        FRorep = obs.calcFrepTotal(robotPos,V)
        FRo = np.add(FRo, FRorep)
        #Plotting
        plt.plot(obs.obstaclePos[0],obs.obstaclePos[1],'o',color='green')
        cir = plt.Circle((obs.obstaclePos[0],obs.obstaclePos[1]),obs.p0,fill=False)
        plt.gca().add_artist(cir)
        obs.advanceStep()
        
    
    robotPos = robotPos + FRo
    print("FRo: " + str(FRo))
    plt.plot(goalPos[0],goalPos[1],'x')
    plt.plot(robotPos[0],robotPos[1],'o',color='red')
    plt.axis('square')
    
    

def animateAFV(frame):
    global robotPos
    obstacles = np.array([[0,0], [20,80]])
    goalPos  = np.array([60,60])

    startPos = np.array([10,15])
    xi = .100
    VG = np.array([0,0])
    V0 = np.array([0,0])
    V = np.array([0,0])
    KV = 60.
    #KV = 60
    KVPrime = 70.

    k=0.04
    i=0
    repAttRatio = 1

    x, y = np.meshgrid(np.arange(0, 100, 10, dtype=float),
                   np.arange(0, 100, 10, dtype=float))

    uatt = calcFatt(xi,calcpG(goalPos,[x,y]),KV,VG,V)[0]
    vatt = calcFatt(xi,calcpG(goalPos,[x,y]),KV,VG,V)[1]
    Fatt = np.multiply([uatt, vatt],(1/repAttRatio))
    print(uatt.shape)

    F = Fatt #np.zeros((10,10))#Fatt
    print("Fatt: " + str(Fatt.shape))
    for i,obsPos in enumerate(obstacles):
        urep = calcFrep(x,y,obsPos,k,repAttRatio)[0]
        vrep = calcFrep(x,y,obsPos,k,repAttRatio)[1]
        #print(F)
        Frep = [urep, vrep]
        F = np.add(F, Frep)
    print("F: " + str(F.shape))
    uRoatt = calcFatt(xi,calcpG(goalPos,[robotPos[0],robotPos[1]]),KV,VG,V)[0]
    vRoatt = calcFatt(xi,calcpG(goalPos,[robotPos[0],robotPos[1]]),KV,VG,V)[1]
    #print("vRoatt: " + str(vRoatt))
    FRoatt = np.multiply([uRoatt, vRoatt],(1/repAttRatio))
    #print("FRoatt: " + str(FRoatt))
    
    FRo = FRoatt
    #print(FRo)
    
    #Frep = [0,0]
    for i,obsPos in enumerate(obstacles):
        uRorep = calcFrep(robotPos[0],robotPos[1], obsPos,k,repAttRatio)[0]
        vRorep = calcFrep(robotPos[0],robotPos[1], obsPos,k,repAttRatio)[1]
        #print(uRorep)
        FRorep = [uRorep, vRorep]
        #print(Frep)
        FRo = np.add(FRo, FRorep)
    
    robotPos = robotPos + FRo

    plt.quiver(x, y, F[0], F[1])
    plt.plot(goalPos[0],goalPos[1],'x')
    plt.plot(robotPos[0],robotPos[1],'o',color='red')
    
        #plt.show()

fig = plt.figure()
ani = FuncAnimation(fig, animateSOC, interval=10,frames=1000)
plt.show()

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
