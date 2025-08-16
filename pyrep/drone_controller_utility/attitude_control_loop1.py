#define wrap_180(x) (x < -M_PI ? x+(2*M_PI) : (x > M_PI ? x - (2*M_PI): x))
#include "attitude_control_loop.h"
from pyrep.drone_controller_utility.PID import PID
from pyrep.drone_controller_utility.controller_utility import ControllerUtility
import math
from pyrep.backend import sim
import numpy as np

class AttitudeController:
    def __init__(self):
        #General parameters
        self.last_time = sim.simGetSimulationTime() #seconds
        self.last_time1 = sim.simGetSimulationTime() #seconds

        self.controller_utility = ControllerUtility()
        self.roll_PID = PID()
        self.pitch_PID = PID() 
        self.yaw_PID = PID()
        self.roll_rate_PID = PID()
        self.pitch_rate_PID = PID() 
        self.yaw_rate_PID = PID()

        # angular loop PID
        roll_PID_params = [1, 4.5, 0, 0, 0] #1, 4.5, 0, 0, 0
        self.roll_PID.SetGainParameters(roll_PID_params)

        pitch_PID_params = [1, 4.5, 0, 0, 0] #1, 4.5, 0, 0, 0
        self.pitch_PID.SetGainParameters(pitch_PID_params)

        yaw_PID_params = [1, 0.5, 0, 0.1, 0] #1 1 0 0 0
        self.yaw_PID.SetGainParameters(yaw_PID_params)

        # angular rate loop PID
        roll_rate_PID_params = [1, 11.5, 0, 3.8, 0] #1, 11.5, 0, 3.8, 0
        self.roll_rate_PID.SetGainParameters(roll_rate_PID_params)

        pitch_rate_PID_params = [1, 11.5, 0, 3.8, 0] #1, 11.5, 0, 3.8, 0
        self.pitch_rate_PID.SetGainParameters(pitch_rate_PID_params)

        yaw_rate_PID_params = [1, 0.5, 0, 0.1, 0] #1 5 0 0 0
        self.yaw_rate_PID.SetGainParameters(yaw_rate_PID_params)

        self.KT = 0.07
        self.Kd = 0.0139
        self.l = 20

        self.motor_lim = 89*89

        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.w4 = 0


    def CalculateAttitudeControl(self,control_cmd_input,euler):
        #Convert quaternion to Euler angles
        meas_roll = euler[0]
        meas_pitch = euler[1]
        meas_yaw = euler[2]

        #Get simulator time
        current_time = sim.simGetSimulationTime()
        dt = current_time - self.last_time+0.01
        self.last_time = current_time

        if dt == 0.0:
            return
        #control_cmd_input: roll,pitch,yaw_rate,thrust
        p_des = self.roll_PID.ComputeCorrection(control_cmd_input[0], meas_roll, dt)
        q_des = self.pitch_PID.ComputeCorrection(control_cmd_input[1], meas_pitch, dt)
        r_des = - self.yaw_PID.ComputeCorrectionLimit(control_cmd_input[2], meas_yaw, dt)

        r_des = self.controller_utility.limit(r_des, -50.0 * np.pi / 180.0, 50.0 * np.pi / 180.0)

        desired_angular_rates = np.zeros([4])
        desired_angular_rates[0] = control_cmd_input[3]
        desired_angular_rates[1] = p_des
        desired_angular_rates[2] = q_des
        desired_angular_rates[3] = r_des

        #print(control_cmd_input[0],control_cmd_input[1],control_cmd_input[2],control_cmd_input[3])
        return desired_angular_rates

        
    def CalculateRateControl(self, des_rate_input, euler, angular_velocity):
        meas_roll = euler[0]
        meas_pitch = euler[1]
        meas_yaw = euler[2]

        #Get simulator time
        current_time = sim.simGetSimulationTime()
        dt = current_time - self.last_time1+0.01
        self.last_time1 = current_time

        if dt == 0.0:
            return

        U1 = des_rate_input[0]
        U2 = self.roll_rate_PID.ComputeCorrection(des_rate_input[1],angular_velocity[0], dt)
        U3 = self.pitch_rate_PID.ComputeCorrection(des_rate_input[2],angular_velocity[1], dt)
        U4 = self.yaw_rate_PID.ComputeCorrection(des_rate_input[3], -angular_velocity[2], dt)
        
        desired_control_cmds = np.zeros([4])
        desired_control_cmds[0] = U1
        desired_control_cmds[1] = U2
        desired_control_cmds[2] = U3
        desired_control_cmds[3] = U4

        return desired_control_cmds


    def CalculateMotorCommands(self,control_inputs):

        U1 = control_inputs[0]
        U2 = control_inputs[1]
        U3 = control_inputs[2]
        U4 = control_inputs[3]

        #Control input to motor mapping  
        # 十字形混控器
        #w1 = U1/(4*KT) - U3/(2*KT*l) + U4/(4*Kd)
        #w2 = U1/(4*KT) - U2/(2*KT*l) - U4/(4*Kd)
        #w3 = U1/(4*KT) + U3/(2*KT*l) + U4/(4*Kd)
        #w4 = U1/(4*KT) + U2/(2*KT*l) - U4/(4*Kd)

        # x字型混控器
        w1 = U1/(4*self.KT) + U2/(2*self.KT*self.l) -U3/(2*self.KT*self.l) + U4/(4*self.Kd)
        w2 = U1/(4*self.KT) + U2/(2*self.KT*self.l) +U3/(2*self.KT*self.l) - U4/(4*self.Kd)
        w3 = U1/(4*self.KT) - U2/(2*self.KT*self.l) +U3/(2*self.KT*self.l) + U4/(4*self.Kd)
        w4 = U1/(4*self.KT) - U2/(2*self.KT*self.l) -U3/(2*self.KT*self.l) - U4/(4*self.Kd)

        #w1 = U1/(4*KT) + U2/(2*Kl) -U3/(2*Kl) + U4/(4*Kd)
        #w2 = U1/(4*KT) + U2/(2*Kl) +U3/(2*Kl) - U4/(4*Kd)
        #w3 = U1/(4*KT) - U2/(2*Kl) +U3/(2*Kl) + U4/(4*Kd)
        #w4 = U1/(4*KT) - U2/(2*Kl) -U3/(2*Kl) - U4/(4*Kd)

        # Limit Based on Motor Parameters
        w1 = self.controller_utility.limit(w1, 0, self.motor_lim)
        w2 = self.controller_utility.limit(w2, 0, self.motor_lim)
        w3 = self.controller_utility.limit(w3, 0, self.motor_lim)
        w4 = self.controller_utility.limit(w4, 0, self.motor_lim)

        #Calculate motor speeds
        w1 = np.sqrt(w1)
        w2 = np.sqrt(w2)
        w3 = np.sqrt(w3)
        w4 = np.sqrt(w4)  

        desired_motor_velocities = np.zeros([4])
        desired_motor_velocities[0] = w1
        desired_motor_velocities[1] = w2
        desired_motor_velocities[2] = w3
        desired_motor_velocities[3] = w4

        return desired_motor_velocities



