import numpy as np
class APFObstacle:
    def __init__(self,obstaclePos,goalPos,p0,V0):
            self.goalPos = goalPos
            self.V0 = V0
            self.p0 = p0
            self.obstaclePos = obstaclePos   
            self.nu = 80
            self.n = 2
            self.KV = 60
            self.KVPrime = 70   

    def calcFrep1(self,currentPos):
        pQ = self.calcpQ(currentPos)
        Frep1 = self.nu*(pQ**-1 - self.p0**-1)*((pQ**-1)**2)*np.power((currentPos-self.goalPos),self.n)
        return Frep1
    
    def calcFrep2(self,currentPos):
        pQ = self.calcpQ(currentPos)
        Frep2 = (self.n/2)*self.nu*((pQ**-1 - self.p0**-1)**2)*np.power((currentPos-self.goalPos),self.n-1)
        return Frep2
    
    def calcFrepTotal(self,currentPos,V):
        pQ = self.calcpQ(currentPos)
        Frep1 = self.calcFrep1(currentPos)
        Frep2 = self.calcFrep2(currentPos)

        if pQ <= self.p0:
            Frep = np.add(np.add(Frep1, Frep2),self.KVPrime*(V-self.V0))
        else:
            Frep = self.KVPrime*(V - self.V0)
        return Frep

    def calcpQ(self,currentPos):
        pQ = np.linalg.norm(currentPos-self.obstaclePos)
        return pQ
    def calcpG(self,currentPos):
        pG = pG = currentPos-self.goalPos
        return pG