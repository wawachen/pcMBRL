from pyrep.robots.mobiles.turtlebot import TurtleBot
from pyrep.robots.mobiles.turtlebot_original import TurtleBot_o
import numpy as np
from pyrep.envs.array_specs import ArraySpec

class Turtle:
    def __init__(self, id):
        self.agent = TurtleBot(id)

    def get_2d_pos(self):
        return self.agent.get_position()[:2]

    def get_heading(self):
        return self.agent.get_orientation()[2]

    def is_crash(self):
        return self.agent.check_collision() # check whether collide with all objects
        
    # def _reset(self):
    #     p = np.random.random.uniform(-2.5,2.5)
    #     self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])

    def set_action(self, action):
        #Turtlebot is controlled by angular velocity in z axis and linear velocity in x axis
        actuation = self.agent.move(action)
        self.agent.set_joint_target_velocities(actuation)

from pyrep.drone_controller_utility.turtlebot_position_controller import TurtlebotPositionController

class Turtle_o:
    def __init__(self, id):
        self.agent = TurtleBot_o(id)
        self.position_controller = TurtlebotPositionController()
        self.goal = np.zeros(2)

    def get_2d_pos(self):
        return self.agent.get_position()[:2]
    
    def get_2d_local_vel(self):
        return np.array(self.agent.get_base_velocities()[:2])

    def get_heading(self):
        return self.agent.get_orientation()[2]
    
    def get_orientation(self):
        return self.agent.get_orientation()

    def is_crash(self):
        return self.agent.check_collision() # check whether collide with all objects
    
    def reset_controller(self):
        self.position_controller.reset()
    
    def calculate_position_control(self, other_robots_pos=None):
        """
        Calculate position control with optional collision avoidance
        
        Args:
            other_robots_pos: list of [x, y] positions of other robots (optional)
            
        Returns:
            [angular_vel, linear_vel] control commands
        """
        current_pos = self.get_2d_pos()
        current_heading = self.get_heading()
        
        # Calculate control commands with collision avoidance
        linear_vel, angular_vel = self.position_controller.calculate_position_control_with_avoidance(
            [self.goal[0], self.goal[1]], current_pos, current_heading, other_robots_pos
        )

        return [angular_vel, linear_vel]
    
    def set_goal(self, goal_pos):
        """Set the goal position for the robot"""
        self.goal[0] = goal_pos[0]
        self.goal[1] = goal_pos[1]

    def set_action(self, action):
        #Turtlebot is controlled by angular velocity in z axis and linear velocity in x axis
        actuation = self.agent.move(action)
        self.agent.set_joint_target_velocities(actuation)

   
            
        
