from pyrep.drone_controller_utility.PID import PID
from pyrep.drone_controller_utility.controller_utility import ControllerUtility
import math
from pyrep.backend import sim
import numpy as np


class VelocityController:
  def __init__(self):
    self.x_vel_PID = PID()
    self.y_vel_PID = PID()
    self.z_vel_PID = PID()
    self.controller_utility = ControllerUtility()
    #self.last_time = sim.simGetSystemTimeInMs(-1) #seconds

    self.last_time = sim.simGetSimulationTime()

    x_vel_PID_params = [1, 2, 0, 0, 0]  #2, 0, 0
    self.x_vel_PID.SetGainParameters(x_vel_PID_params)

    y_vel_PID_params = [1, 2, 0, 0, 0]
    self.y_vel_PID.SetGainParameters(y_vel_PID_params)

    z_vel_PID_params = [1, 5, 0, 0, 0]
    self.z_vel_PID.SetGainParameters(z_vel_PID_params)


  def CalculateVelocityControl(self, vel_, euler, twist):

    #Convert quaternion to Euler angles
    gps_roll = euler[0]
    gps_pitch = euler[1]
    gps_yaw = euler[2]

    sin_yaw = math.sin(gps_yaw)
    cos_yaw = math.cos(gps_yaw)
    # Get simulator time
    current_time = sim.simGetSimulationTime()
    dt = current_time - self.last_time+0.01
    self.last_time = current_time

    #dt = sim.simGetSystemTimeInMs(self.last_time)/1000
    #print(dt)
    if dt == 0.0:
      return

    gps_vel_x = twist[0]
    gps_vel_y = twist[1]
    gps_vel_z = twist[2]

    x_acc_des = self.x_vel_PID.ComputeCorrection(vel_[0], twist[0], dt)
    y_acc_des = self.y_vel_PID.ComputeCorrection(vel_[1], twist[1], dt)
    z_acc_des = self.z_vel_PID.ComputeCorrection(vel_[2], twist[2], dt)

    x_acc_des = self.controller_utility.limit(x_acc_des, -1, 1)
    y_acc_des = self.controller_utility.limit(y_acc_des, -1, 1)
    #z_acc_des = controller_utility.limit(z_acc_des, -1, 1);

    des_acc_cmds = np.zeros([4])
    des_acc_cmds[0] = x_acc_des
    des_acc_cmds[1] = y_acc_des
    des_acc_cmds[2] = z_acc_des
    des_acc_cmds[3] = vel_[3]

    return des_acc_cmds
    


