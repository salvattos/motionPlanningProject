import numpy as np
class Goal:
    def __init__(self,goalPos,VG,xi):
            self.VG = VG
            self.xi = xi
            self.goalPos = goalPos
            self.KV = .9
    def calcFatt(self,robotPos,V):
        pG = self.calcpG(robotPos)
        u = -self.xi * pG[0] + (self.KV * (V-self.VG))[0] 
        v = -self.xi * pG[1] + (self.KV * (V-self.VG))[1]
        Fatt = [u,v]
        return Fatt 
    
    def calcpG(self,robotPos):
        u = robotPos[0] - self.goalPos[0]
        v = robotPos[1] - self.goalPos[1]
        pG = [u,v]
        return pG

    def advanceStep(self):
        self.goalPos += + self.VG