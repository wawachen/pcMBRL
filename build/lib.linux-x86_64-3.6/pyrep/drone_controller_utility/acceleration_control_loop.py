#include "acceleration_control_loop.h"
#define GRAVITY 9.81
from pyrep.drone_controller_utility.PID import PID
from pyrep.drone_controller_utility.controller_utility import ControllerUtility
import math
from pyrep.backend import sim
import numpy as np

class AccelerationController:
    def __init__(self):
        self.GRAVITY = 9.81
        self.controller_utility = ControllerUtility()
        self.x_acc_PID = PID()
        self.y_acc_PID = PID()
        self.z_acc_PID = PID()

        self.last_time = sim.simGetSimulationTime() #seconds
        x_acc_PID_params = [1, 1, 0, 0, 0]
        self.x_acc_PID.SetGainParameters(x_acc_PID_params)

        y_acc_PID_params = [1, 1, 0, 0, 0]
        self.y_acc_PID.SetGainParameters(y_acc_PID_params)

        z_acc_PID_params = [1, 0.8, 0.01, 0.1, 200] #1, 0.8, 0.01, 1, 200
        self.z_acc_PID.SetGainParameters(z_acc_PID_params)


    def CalculateAccelerationControl(self, cmd_acc, cmd_yawrate, euler,linear_acceleration):
        gps_roll = euler[0]
        gps_pitch = euler[1]
        gps_yaw = euler[2]

        #ROS_DEBUG("RPY = (%lf, %lf, %lf)", gps_roll, gps_pitch, gps_yaw);
        sin_yaw = math.sin(gps_yaw)
        cos_yaw = math.cos(gps_yaw)

        # Get simulator time
        current_time = sim.simGetSimulationTime()
        dt = current_time - self.last_time+0.01
        self.last_time = current_time
        if dt == 0.0:
            return

        # calculate the acceleration to base frame
        # acc: x,y,z,
        x_base_acc  =  cos_yaw * cmd_acc[0]  + sin_yaw * cmd_acc[1]
        y_base_acc =  - sin_yaw * cmd_acc[0]  + cos_yaw * cmd_acc[1]
        z_base_acc = cmd_acc[2]

        #this value is equal to 7.9426 in this simulator under this mixer;

        roll_des = - self.controller_utility.limit(math.atan(y_base_acc / self.GRAVITY), -0.26, 0.26)
        pitch_des = self.controller_utility.limit(math.atan(x_base_acc / self.GRAVITY), -0.26, 0.26)
        yaw_des = cmd_yawrate

        thrust_des = self.z_acc_PID.ComputeCorrection(z_base_acc, linear_acceleration[2] + self.GRAVITY, dt)
        thrust_des = thrust_des + 7.9426

        #roll,pitch,yaw,thrust
        des_attitude_cmds = np.zeros([4])
        des_attitude_cmds[0] = roll_des	
        des_attitude_cmds[1] = pitch_des
        des_attitude_cmds[2] = yaw_des
        des_attitude_cmds[3] = thrust_des

        return des_attitude_cmds

    


