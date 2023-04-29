import numpy as np
class Agent:
    def __init__(self,robotPos,V):
            self.V = V
            self.robotPos = robotPos
    def applyForce(self,Fro):
        self.robotPos += Fro