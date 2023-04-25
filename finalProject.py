import numpy as np
from APFObstacle import APFObstacle

def calcFatt(xi,pG,KV,VG,V):
    Fatt = -xi*pG + (KV*(VG-V))
    return Fatt


def calcpG(goalPos,currentPos):
    pG = currentPos-goalPos #might need to switch order
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
FattList = []
for x in range(0,100):
    for y in range(0,100):
        currentPos = np.array([x,y])
        #Attraction force
        Fatt = calcFatt(xi,calcpG(goalPos,currentPos),KV,VG,V)
        FattList.append(Fatt)
        #Frep = OB1.calcFrepTotal(currentPos,V)
        Frep = 0
        Fsum = Frep+Fatt

print(FattList)