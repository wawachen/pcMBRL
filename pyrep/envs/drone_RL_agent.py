from pyrep.robots.mobiles.RL_quadricopter import RLQuadricopter
from pyrep.robots.mobiles.RL_quadricopter import RLQuadricopter_nc
from pyrep.robots.mobiles.RL_quadricopter import RLQuadricopter_nc1
from pyrep.robots.mobiles.RL_quadricopter import RLQuadricopter_wc
# from pyrep.robots.mobiles.new_quadricopter import NewQuadricopter
import numpy as np
from pyrep.envs.array_specs import ArraySpec
from pyrep.drone_controller_utility.PID import PID
from rvo2 import ORCA_agent as orca_agent
import cv2

#with_camera
class Drone:
    def __init__(self, id):
        self.agent = RLQuadricopter(id,4)
        self.wall_collision = None

        self.gains = np.array([1, 1, 0, 0, 0])
        self.level_pid = PID()
        self.level_pid.SetGainParameters(self.gains)
        # self.level = 1.7

    def crash_detection(self):
        if self.agent.get_drone_position()[2] < 1.3:
            return 1
        else: 
            return 0

    def get_2d_pos(self):
        return self.agent.get_drone_position()[:2]
    
    def set_2d_pose(self, pos):
        """Sets the 2D (top-down) pose of the robot [x, y, z]
        :param pose: A List containing the x, y, yaw (in radians).
        """
        x, y, z = pos
        self.agent.set_position([x, y, z])
        self.agent.set_orientation([0, 0, 0])

    def get_2d_vel(self):
        return self.agent.get_velocities()[0][:2]

    def get_rgb_image(self):
        image = self.agent.get_panoramic()[0]
        image = image[57:157,:,(2,1,0)] #107:157
        # image = image[47:107,:32,(2,1,0)] #107:157
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # print(image)
        # cv2.imshow("Original image",image)
        # print(image.dtype)
        return image

    def get_depth_image(self):
        #not working
        image = self.agent.get_panoramic()[1]

        # image = image[:,:]
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        print(image.shape)
        cv2.imshow("Original image",image)

    def get_heading(self):
        return self.agent.get_drone_orientation()[2]

    def _reset(self):
        #self.suction_cup.release()
        self.agent.drone_reset()
        # p = np.random.random.uniform()
        # self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])

    def hover(self,height):
        p_pos = self.agent.get_drone_position()

        bac_vel = self.level_pid.ComputeCorrection(height, p_pos[2], 0.01)

        vels = self.agent.velocity_controller1(0,0,bac_vel)

        self.agent.set_propller_velocity(vels)

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action(self, action):
        #vx, vy
        vels = self.agent.velocity_controller1(action[0],action[1])
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    # def set_action_vel(self,action):
    #     act = np.array([action[0],action[1]])
    #     vels = self.agent.position_controller(act)
    #     self.agent.set_propller_velocity(vels[:])

    #     self.agent.control_propeller_thrust(1)
    #     self.agent.control_propeller_thrust(2)
    #     self.agent.control_propeller_thrust(3)
    #     self.agent.control_propeller_thrust(4)

#scratch
class Drone_s:
    def __init__(self, id):
        self.agent = RLQuadricopter_nc(id,4)
        self.wall_collision = None

        self.gains = np.array([1, 1, 0, 0, 0])
        self.level_pid = PID()
        self.level_pid.SetGainParameters(self.gains)
        # self.level = 1.7

    def crash_detection(self):
        if self.agent.get_drone_position()[2] < 1.3:
            return 1
        else: 
            return 0

    def get_2d_pos(self):
        return self.agent.get_drone_position()[:2]

    def set_2d_pose(self, pos):
        """Sets the 2D (top-down) pose of the robot [x, y, z]
        :param pose: A List containing the x, y, yaw (in radians).
        """
        x, y, z = pos
        self.agent.set_position([x, y, z])
        self.agent.set_orientation([0, 0, 0])

    def get_2d_vel(self):
        return self.agent.get_velocities()[0][:2]

    def get_heading(self):
        return self.agent.get_drone_orientation()[2]

    def _reset(self):
        #self.suction_cup.release()
        self.agent.drone_reset()
        # p = np.random.random.uniform()
        # self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])

    def hover(self,height):
        p_pos = self.agent.get_drone_position()

        bac_vel = self.level_pid.ComputeCorrection(height, p_pos[2], 0.01)

        vels = self.agent.velocity_controller1(0,0,bac_vel)

        self.agent.set_propller_velocity(vels)

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action(self, action):
        #vx, vy
        vels = self.agent.velocity_controller1(action[0],action[1])
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)
    
    def set_discrete_action(self, action):
        if action == 0:
           vels = self.agent.velocity_controller1(0.0,0.0)
        if action == 1:
           vels = self.agent.velocity_controller1(-0.5,0.0) 
        if action == 2:
           vels = self.agent.velocity_controller1(0.5,0.0)
        if action == 3:
           vels = self.agent.velocity_controller1(0.0,-0.5)
        if action == 4:
           vels = self.agent.velocity_controller1(0.0,0.5)

        self.agent.set_propller_velocity(vels[:])
        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)
    
    def set_discrete_action1(self, action):
        if action == 0:
           vels = self.agent.velocity_controller1(0.0,0.0)
        if action == 1:
           vels = self.agent.velocity_controller1(-0.5,0.0) 
        if action == 2:
           vels = self.agent.velocity_controller1(0.5,0.0)
        if action == 3:
           vels = self.agent.velocity_controller1(0.0,-0.5)
        if action == 4:
           vels = self.agent.velocity_controller1(0.0,0.5)
        if action == 5:
           vels = self.agent.velocity_controller1(0.5,0.5)
        if action == 6:
           vels = self.agent.velocity_controller1(-0.5,0.5)
        if action == 7:
           vels = self.agent.velocity_controller1(0.5,-0.5)
        if action == 8:
           vels = self.agent.velocity_controller1(-0.5,-0.5)
        
        self.agent.set_propller_velocity(vels[:])
        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action_pos(self,action):
        act = np.array([action[0],action[1],1.7])
        vels = self.agent.position_controller(act)
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

# with proximty sensors
class Drone_s1:
    def __init__(self, id):
        self.agent = RLQuadricopter_nc1(id,4)
        self.wall_collision = None

        self.gains = np.array([1, 1, 0, 0, 0])
        self.level_pid = PID()
        self.level_pid.SetGainParameters(self.gains)
        # self.level = 1.7

    def crash_detection(self):
        if self.agent.get_drone_position()[2] < 1.3:
            return 1
        else: 
            return 0

    def get_2d_pos(self):
        return self.agent.get_drone_position()[:2]

    def set_2d_pose(self, pos):
        """Sets the 2D (top-down) pose of the robot [x, y, z]
        :param pose: A List containing the x, y, yaw (in radians).
        """
        x, y, z = pos
        self.agent.set_position([x, y, z])
        self.agent.set_orientation([0, 0, 0])

    def get_2d_vel(self):
        return self.agent.get_velocities()[0][:2]
    
    def get_promity_readings(self):
        return self.agent.get_proximity()

    def get_heading(self):
        return self.agent.get_drone_orientation()[2]

    def _reset(self):
        #self.suction_cup.release()
        self.agent.drone_reset()
        # p = np.random.random.uniform()
        # self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])

    def hover(self,height):
        p_pos = self.agent.get_drone_position()

        bac_vel = self.level_pid.ComputeCorrection(height, p_pos[2], 0.01)

        vels = self.agent.velocity_controller1(0,0,bac_vel)

        self.agent.set_propller_velocity(vels)

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action(self, action):
        #vx, vy
        vels = self.agent.velocity_controller1(action[0],action[1])
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)
    
    def set_discrete_action(self, action):
        if action == 0:
           vels = self.agent.velocity_controller1(0.0,0.0)
        if action == 1:
           vels = self.agent.velocity_controller1(-0.5,0.0) 
        if action == 2:
           vels = self.agent.velocity_controller1(0.5,0.0)
        if action == 3:
           vels = self.agent.velocity_controller1(0.0,-0.5)
        if action == 4:
           vels = self.agent.velocity_controller1(0.0,0.5)

        self.agent.set_propller_velocity(vels[:])
        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)
    
    def set_discrete_action1(self, action):
        if action == 0:
           vels = self.agent.velocity_controller1(0.0,0.0)
        if action == 1:
           vels = self.agent.velocity_controller1(-0.5,0.0) 
        if action == 2:
           vels = self.agent.velocity_controller1(0.5,0.0)
        if action == 3:
           vels = self.agent.velocity_controller1(0.0,-0.5)
        if action == 4:
           vels = self.agent.velocity_controller1(0.0,0.5)
        if action == 5:
           vels = self.agent.velocity_controller1(0.5,0.5)
        if action == 6:
           vels = self.agent.velocity_controller1(-0.5,0.5)
        if action == 7:
           vels = self.agent.velocity_controller1(0.5,-0.5)
        if action == 8:
           vels = self.agent.velocity_controller1(-0.5,-0.5)
        
        self.agent.set_propller_velocity(vels[:])
        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action_pos(self,action):
        act = np.array([action[0],action[1],1.7])
        vels = self.agent.position_controller(act)
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

#############################################################
#drone sensor
class Drone_s_w:
    def __init__(self, id):
        self.agent = RLQuadricopter_wc(id,4)
        self.wall_collision = None

        self.gains = np.array([1, 1, 0, 0, 0])
        self.level_pid = PID()
        self.level_pid.SetGainParameters(self.gains)
        # self.level = 1.7

    def crash_detection(self):
        if self.agent.get_drone_position()[2] < 1.3:
            return 1
        else: 
            return 0

    def get_2d_pos(self):
        return self.agent.get_drone_position()[:2]

    def get_goals_sensor(self, goals):
        x = self.agent.get_drone_position()[0]
        y = self.agent.get_drone_position()[1]
        z = self.agent.get_drone_position()[2]
        d_list = []

        for i in range(goals.shape[0]):
            d = np.sqrt((goals[i,0]-x)**2 + (goals[i,1]-y)**2 + (goals[i,2]-z)**2)
            d_list.append(d)

        return d_list

    def get_neighbour_pos_from_camera(self):
        pos = self.agent.get_neighbour_pos_from_camera()
        return pos

    def set_2d_pose(self, pos):
        """Sets the 2D (top-down) pose of the robot [x, y, z]
        :param pose: A List containing the x, y, yaw (in radians).
        """
        x, y, z = pos
        self.agent.set_position([x, y, z])
        self.agent.set_orientation([0, 0, 0])

    def get_2d_vel(self):
        return self.agent.get_velocities()[0][:2]

    def get_heading(self):
        return self.agent.get_drone_orientation()[2]

    def _reset(self):
        #self.suction_cup.release()
        self.agent.drone_reset()
        # p = np.random.random.uniform()
        # self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])

    def hover(self,height):
        p_pos = self.agent.get_drone_position()

        bac_vel = self.level_pid.ComputeCorrection(height, p_pos[2], 0.01)

        vels = self.agent.velocity_controller1(0,0,bac_vel)

        self.agent.set_propller_velocity(vels)

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action(self, action):
        #vx, vy
        vels = self.agent.velocity_controller1(action[0],action[1])
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action_pos(self,action):
        act = np.array([action[0],action[1],1.7])
        vels = self.agent.position_controller(act)
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

##############################################################
#for discrete action
class Drone_s_dis:
    def __init__(self, id):
        self.agent = RLQuadricopter_nc(id,4)
        self.wall_collision = None

        self.gains = np.array([1, 1, 0, 0, 0])
        self.level_pid = PID()
        self.level_pid.SetGainParameters(self.gains)
        # self.level = 1.7

    def crash_detection(self):
        if self.agent.get_drone_position()[2] < 1.3:
            return 1
        else: 
            return 0

    def get_2d_pos(self):
        return self.agent.get_drone_position()[:2]
    
    def set_2d_pose(self, pos):
        """Sets the 2D (top-down) pose of the robot [x, y, z]
        :param pose: A List containing the x, y, yaw (in radians).
        """
        x, y, z = pos
        self.agent.set_position([x, y, z])
        self.agent.set_orientation([0, 0, 0])

    def get_2d_vel(self):
        return self.agent.get_velocities()[0][:2]

    def get_heading(self):
        return self.agent.get_drone_orientation()[2]

    def _reset(self):
        #self.suction_cup.release()
        self.agent.drone_reset()
        # p = np.random.random.uniform()
        # self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])

    def hover(self,height):
        p_pos = self.agent.get_drone_position()

        bac_vel = self.level_pid.ComputeCorrection(height, p_pos[2], 0.01)

        vels = self.agent.velocity_controller1(0,0,bac_vel)

        self.agent.set_propller_velocity(vels)

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action(self, action):
        act = np.where(action==1)[0][0]
        # print(act)
        #vx, vy
        if act == 0:    
            vels = self.agent.velocity_controller1(-0.5,0.0)
            self.agent.set_propller_velocity(vels[:])
        if act == 1:
            vels = self.agent.velocity_controller1(0.5,0.0)
            self.agent.set_propller_velocity(vels[:])
        if act == 2:
            vels = self.agent.velocity_controller1(0.0,-0.5)
            self.agent.set_propller_velocity(vels[:])
        if act == 3:
            vels = self.agent.velocity_controller1(0.0,0.5)
            self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    # def set_action_pos(self,action):
    #     act = np.array([action[0],action[1],1.7])
    #     vels = self.agent.position_controller(act)
    #     self.agent.set_propller_velocity(vels[:])

    #     self.agent.control_propeller_thrust(1)
    #     self.agent.control_propeller_thrust(2)
    #     self.agent.control_propeller_thrust(3)
    #     self.agent.control_propeller_thrust(4)
##############################################################
            
#ORCA+RL
class Drone_ORCA:
    def __init__(self, id, num_agent):
        #orca parameter
        self.neighborDist = 10
        if num_agent == 3:
            self.maxNeighbors = 2
        elif num_agent==6:
            self.maxNeighbors = 5
        elif num_agent==4:
            self.maxNeighbors = 3

        self.timeHorizon = 4.5
        self.radius = 0.3
        self.maxSpeed = 1.0
        #----------------------------------
        self.agent = RLQuadricopter_nc(id,4)
        self.wall_collision = None

        self.gains = np.array([1, 1, 0, 0, 0])
        self.level_pid = PID()
        self.level_pid.SetGainParameters(self.gains)
        # self.level = 1.7

    def get_agent(self, pos):
        self.orca_agent = orca_agent(pos, self.neighborDist,
                 self.maxNeighbors, self.timeHorizon,
                 self.radius, self.maxSpeed,
                 (0, 0))

    def update_agent(self):
        m = self.get_2d_pos()
        m1 = self.get_2d_vel()
        position = (m[0],m[1])
        velocity = (m1[0],m1[1])
        self.orca_agent.self_update(position,velocity)

    def set_prefer_velocity(self,vel):
        self.orca_agent.setAgentPrefVelocity(vel)

    def computeNeighbors(self, positions_n, velocities_n):
        self.orca_agent.update_neighbour_states(np.array(positions_n),np.array(velocities_n))
        
    def computeNewVelocity(self):
        self.orca_agent.computeNewVelocity()

    def computeNewVelocity1(self):
        return self.orca_agent.computeNewVelocity1()

    def crash_detection(self):
        if self.agent.get_drone_position()[2] < 1.3:
            return 1
        else: 
            return 0

    def get_planes(self):
        return self.orca_agent.get_orca_lines()

    def get_2d_pos(self):
        return self.agent.get_drone_position()[:2]

    def get_2d_vel(self):
        return self.agent.get_velocities()[0][:2]

    def get_heading(self):
        return self.agent.get_drone_orientation()[2]

    def _reset(self):
        #self.suction_cup.release()
        self.agent.drone_reset()
        # p = np.random.random.uniform()
        # self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])

    def hover(self,height):
        p_pos = self.agent.get_drone_position()

        bac_vel = self.level_pid.ComputeCorrection(height, p_pos[2], 0.01)

        vels = self.agent.velocity_controller1(0,0,bac_vel)

        self.agent.set_propller_velocity(vels)

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action(self, action):
        #vx, vy
        vels = self.agent.velocity_controller1(action[0],action[1])
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)


#only ORCA
from rvo2 import ORCA_agent as orca_agent
class Drone_ORCA1:
    def __init__(self, id, num_agent):
        #orca parameter
        self.neighborDist = 10
        if num_agent==3:
            self.maxNeighbors = 2 #9
        elif num_agent==6:
            self.maxNeighbors = 5
        elif num_agent==4:
            self.maxNeighbors = 3 

        self.timeHorizon = 4.5  #for 3,6 Uavs: 4.5//14.5
        self.radius = 0.3  #for 3,6 UAVs: 0.3//  
        self.maxSpeed = 1.0
        self.goal = np.zeros(2)
        #----------------------------------
        self.agent = RLQuadricopter_nc(id,4)
        self.wall_collision = None

        self.gains = np.array([1, 1, 0, 0, 0])
        self.level_pid = PID()
        self.level_pid.SetGainParameters(self.gains)
        # self.level = 1.7

    def get_agent(self, pos):
        self.orca_agent = orca_agent(pos, self.neighborDist,
                 self.maxNeighbors, self.timeHorizon,
                 self.radius, self.maxSpeed,
                 (0, 0))

    def update_agent(self):
        m = self.get_2d_pos()
        m1 = self.get_2d_vel()
        position = (m[0],m[1])
        velocity = (m1[0],m1[1])
        self.orca_agent.self_update(position,velocity)

        prefer_vel = self.goal-m
        vel_len = np.sqrt(prefer_vel[0]**2+prefer_vel[1]**2)
        if vel_len>1.0:
          prefer_vel = (prefer_vel/vel_len)*1.0
        self.set_prefer_velocity((prefer_vel[0],prefer_vel[1]))

    def set_prefer_velocity(self,vel):
        self.orca_agent.setAgentPrefVelocity(vel)

    def computeNeighbors(self, positions_n, velocities_n):
        self.orca_agent.update_neighbour_states(np.array(positions_n),np.array(velocities_n))
        
    def computeNewVelocity(self):
        self.orca_agent.computeNewVelocity()
    
    def computeNewVelocity1(self):
        return self.orca_agent.computeNewVelocity1()

    def crash_detection(self):
        if self.agent.get_drone_position()[2] < 1.3:
            return 1
        else: 
            return 0

    def get_planes(self):
        return self.orca_agent.get_orca_lines()

    def get_2d_pos(self):
        return self.agent.get_drone_position()[:2]

    def get_2d_vel(self):
        return self.agent.get_velocities()[0][:2]

    def get_heading(self):
        return self.agent.get_drone_orientation()[2]
    
    def get_orientation(self):
        return self.agent.get_drone_orientation()

    def _reset(self):
        #self.suction_cup.release()
        self.agent.drone_reset()
        # p = np.random.random.uniform()
        # self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])

    def hover(self,height):
        p_pos = self.agent.get_drone_position()

        bac_vel = self.level_pid.ComputeCorrection(height, p_pos[2], 0.01)

        vels = self.agent.velocity_controller1(0,0,bac_vel)

        self.agent.set_propller_velocity(vels)

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)

    def set_action(self, action):
        #vx, vy
        vels = self.agent.velocity_controller1(action[0],action[1])
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)