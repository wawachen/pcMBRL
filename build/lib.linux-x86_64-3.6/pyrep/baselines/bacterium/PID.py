import numpy as np

class PID:
    def __init__(self):
        self.G = 0.0
        self.Kp = 0.0
        self.Kd = 0.0 
        self.Ki = 0.0 
        self.Ki_max= 0.0
        self.speedCmd = 0.0

        self.posErrorPrev = 0.0
        self.posIntegrator = 0.0

        self.error = 0.0
        self.errorPrev = 0.0
        self.integrator = 0.0
    
    def SetGainParameters(self,params):
        self.G = params[0]
        self.Kp = params[1]
        self.Ki = params[2]
        self.Kd = params[3]
        self.Ki_max = params[4]

    def ReSet(self):
        self.errorPrev = 0.0
        self.integrator = 0.0

    def wrap_180(self,x):
        if x < -np.pi:
            return x + 2*np.pi
        elif x > np.pi: 
            return x - 2*np.pi
        else:
            return x

    def ComputeCorrection(self, cmd, pos, loopTime):
        correction = 0
        self.error = cmd - pos

        if self.Ki_max != 0:
            if self.integrator >= self.Ki_max:  
                self.integrator = self.Ki_max
            elif self.integrator <= - self.Ki_max:
                self.integrator = - self.Ki_max
            else:
                self.integrator += self.error
        else:
            self.integrator += self.error
        
        correction = self.G * (self.Kp * self.error + self.Ki * self.integrator + self.Kd * (self.error - self.errorPrev)/(loopTime))

        self.errorPrev = self.error
        return correction


    def ComputeCorrectionLimit(self, cmd, pos, loopTime):
        correction = 0
        self.error = cmd - pos
        self.error = self.wrap_180(self.error)
        if self.Ki_max != 0:
            if self.integrator >= self.Ki_max: 
                self.integrator = self.Ki_max
            elif self.integrator <= - self.Ki_max:
                self.integrator = - self.Ki_max
            else:
                self.integrator += self.error
        else: 
            self.integrator += self.error
        
        correction = self.G * (self.Kp * self.error + self.Ki * self.integrator + self.Kd * (self.error - self.errorPrev)/(loopTime))

        self.errorPrev = self.error
        return correction
