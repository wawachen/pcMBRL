from pyrep.drone_controller_utility.PID import PID
from pyrep.drone_controller_utility.controller_utility import ControllerUtility
import math
from pyrep.backend import sim
import numpy as np


class PositionController:
    def __init__(self):
        #General parameters
        self.last_time = sim.simGetSimulationTime() #seconds

        self.controller_utility = ControllerUtility()
        

        self.x_PID = PID()
        self.y_PID= PID()
        self.z_PID= PID() 
        self.yaw_PID= PID()
      

        x_PID_params = [1, 1, 0, 0.02, 0] #1, 1, 0, 0.2, 0
        self.x_PID.SetGainParameters(x_PID_params)

        y_PID_params = [1, 1, 0, 0.02, 0] #1, 1, 0, 0.2, 0
        self.y_PID.SetGainParameters(y_PID_params)

        z_PID_params = [1, 1, 0.02, 0.02, 0] #1, 1, 0.02, 0.2, 0
        self.z_PID.SetGainParameters(z_PID_params)


    def CalculatePositionControl(self, wp, euler, position):

        gps_roll = euler[0]
        gps_pitch = euler[1]
        gps_yaw = euler[2]
        #Get simulator time
        current_time = sim.simGetSimulationTime()
        dt = current_time - self.last_time+0.01
        self.last_time = current_time
    
        if dt == 0.0:
            return

        gps_x = position[0]
        gps_y = position[1]
        gps_z = position[2]

        #x, y, z PID
        x_vel_des = self.x_PID.ComputeCorrection(wp[0], position[0], dt)
        y_vel_des = self.y_PID.ComputeCorrection(wp[1], position[1], dt)
        z_vel_des = self.z_PID.ComputeCorrection(wp[2], position[2], dt)
        
        x_vel_des = self.controller_utility.limit(x_vel_des, -1, 1)
        y_vel_des = self.controller_utility.limit(y_vel_des, -1, 1)
        #z_vel_des = controller_utility_.limit(z_vel_des, -1, 1);

        #Yaw PID
        #In this part, do not calcuate the yaw_rate, sent the yaw information into the attitude control loop
        #and calculate in that loop.
        #des_velocity_cmds: x,y,z,yaw_rate
        des_velocity_cmds = np.zeros([4])

        des_velocity_cmds[0] = x_vel_des	
        des_velocity_cmds[1] = y_vel_des
        des_velocity_cmds[2] = z_vel_des

        #this is not yaw_rate, it is yaw
        #des_velocity_cmds.yaw_rate = yaw_des; sent thiCalculatePositionControls information to the attitude control loop directly
        des_velocity_cmds[3] = wp[3]

        return des_velocity_cmds
        

class PositionController1:
    def __init__(self):
        #General parameters
        self.last_time = sim.simGetSimulationTime() #seconds

        self.controller_utility = ControllerUtility()
        

        self.x_PID = PID()
        self.y_PID= PID()
        self.z_PID= PID() 
        self.yaw_PID= PID()

        x_PID_params = [1, 1, 0, 0.02, 0] #1, 1, 0, 0.2, 0
        self.x_PID.SetGainParameters(x_PID_params)

        y_PID_params = [1, 1, 0, 0.02, 0] #1, 1, 0, 0.2, 0
        self.y_PID.SetGainParameters(y_PID_params)

        z_PID_params = [1, 1, 0, 0, 0] #1, 1, 0.02, 0.2, 0
        self.z_PID.SetGainParameters(z_PID_params)


    def CalculatePositionControl(self, wp, euler, position):

        gps_roll = euler[0]
        gps_pitch = euler[1]
        gps_yaw = euler[2]
        #Get simulator time
        current_time = sim.simGetSimulationTime()
        dt = current_time - self.last_time+0.01
        self.last_time = current_time
    
        if dt == 0.0:
            return

        gps_x = position[0]
        gps_y = position[1]
        gps_z = position[2]

        #x, y, z PID
        x_vel_des = self.x_PID.ComputeCorrection(wp[0], position[0], dt)
        y_vel_des = self.y_PID.ComputeCorrection(wp[1], position[1], dt)
        z_vel_des = self.z_PID.ComputeCorrection(wp[2], position[2], dt)
        
        x_vel_des = self.controller_utility.limit(x_vel_des, -1, 1)
        y_vel_des = self.controller_utility.limit(y_vel_des, -1, 1)
        #z_vel_des = controller_utility_.limit(z_vel_des, -1, 1);

        #Yaw PID
        #In this part, do not calcuate the yaw_rate, sent the yaw information into the attitude control loop
        #and calculate in that loop.
        #des_velocity_cmds: x,y,z,yaw_rate
        des_velocity_cmds = np.zeros([4])

        des_velocity_cmds[0] = x_vel_des	
        des_velocity_cmds[1] = y_vel_des
        des_velocity_cmds[2] = z_vel_des

        #this is not yaw_rate, it is yaw
        #des_velocity_cmds.yaw_rate = yaw_des; sent thiCalculatePositionControls information to the attitude control loop directly
        des_velocity_cmds[3] = wp[3]

        return des_velocity_cmds



