import numpy as np
class APFObstacle:
    def __init__(self,obstaclePos,p0,V0):
            self.V0 = V0
            self.p0 = p0
            self.obstaclePos = obstaclePos   
            self.nu = 50
            self.SOI = p0*2
            self.n = 2
            self.KV = .60
            self.KVPrime = .70   

    def calcFrep1(self,currentPos):
        pQ = self.calcpQ(currentPos)
        Frep1 = -self.nu*(pQ**-1 - self.p0**-1)*((pQ**-1)**2)*np.power((currentPos-self.obstaclePos),self.n)
        return Frep1
    
    def calcFrep2(self,currentPos):
        pQ = self.calcpQ(currentPos)
        Frep2 = (self.n/2)*self.nu*((pQ**-1 - self.p0**-1)**2)*np.power((currentPos-self.obstaclePos),self.n-1)
        return Frep2
    
    def calcFrepTotal(self,currentPos,V):
        pQ = self.calcpQ(currentPos)
        Frep1 = self.calcFrep1(currentPos)
        Frep2 = self.calcFrep2(currentPos)

        if pQ <= self.p0:
            Frep = np.add(np.add(Frep1, Frep2),self.KVPrime*(V-self.V0))
            print("Entered obstacle P0 at point:", str(currentPos))
            print(Frep1)
            print(Frep2)
        elif pQ <= self.SOI:
            #Frep = self.KVPrime*(V - self.V0)
            print("Entered obstacle SOI at point:", str(currentPos))
            u = -((currentPos[1]-self.obstaclePos[1])-(currentPos[0]-self.obstaclePos[0]))#/np.sqrt(sum(pow(-(y-x),2), pow(-(-y-x),2)))
            v = -(-(currentPos[1]-self.obstaclePos[1])-(currentPos[0]-self.obstaclePos[0]))#/np.sqrt(sum(pow(-(y-x),2), pow(-(-y-x),2)))
            if np.arctan2(currentPos[0],self.obstaclePos[0]) < 0:
                Frep = [u,v]#/(magnitude([u,v])))#*(1/(1 + np.exp(k*(magnitude([u,v])-i))))
            else:
                Frep = [-v,-u]
            k = .01
            repAttRatio = 1
            Frep = (Frep/(self.magnitude([u,v])))*1/(1+np.exp(k*(self.magnitude([u,v]))))*(repAttRatio/1)
        else: 
            Frep = 0
        return Frep

    def calcpQ(self,currentPos):
        pQ = np.linalg.norm(currentPos-self.obstaclePos)
        return pQ
    def calcpG(self,currentPos):
        pG = pG = currentPos-self.goalPos
        return pG
    def magnitude(self,vector):
        return np.sqrt(sum(pow(element, 2) for element in vector))
    def advanceStep(self):
        self.obstaclePos += + self.V0