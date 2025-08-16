import numpy as np
import math

class ControllerUtility:
    def __init__(self):
        switchValue = False
        prevInput = False

    def map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


    def limit(self, ind, min, max):
        if ind < min:
          ind = min
        if ind > max:
          ind = max
        return ind

    def GetSwitchValue(self):
        return switchValue

    def UpdateSwitchValue(self, currInput):
        if currInput != prevInput:
            if currInput:
                switchValue = not switchValue
            prevInput = currInput
        
        return switchValue
    

    # def rotateGFtoBF(GF_x, GF_y, GF_z, GF_roll, GF_pitch, GF_yaw):
        
    #     Eigen::Matrix3d Rot

    #     GF_ = [GF_x, GF_y, GF_z]
    #     Eigen::Vector3d BF_

    #     R_roll = np.array([[1, 0, 0], [0, math.cos(GF_roll), -math.sin(GF_roll)], [0, math.sin(GF_roll), math.cos(GF_roll)]])
    #     R_pitch = np.array([[math.cos(GF_pitch), 0 , math.sin(GF_pitch)], [0, 1, 0], [-math.sin(GF_pitch), 0, math.cos(GF_pitch)]]) 
    #     R_yaw = np.array([[math.cos(GF_yaw), -math.sin(GF_yaw), 0], [math.sin(GF_yaw), math.cos(GF_yaw), 0], [0, 0, 1]]) 

    #     Rot = R_yaw * R_pitch * R_roll
    #     BF_ = GF_.transpose() * Rot

    #     return BF_.transpose()

   


