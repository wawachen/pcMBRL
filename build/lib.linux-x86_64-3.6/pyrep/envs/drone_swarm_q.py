import numpy as np
import scipy.io
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import math
from multiagent.utilities import _relative_headings, _shortest_vec, _distances, _relative_headings, _product_difference, _distance_rewards,take_along_axis

class QScenario(BaseScenario):
    def make_world(self,num):
        world = World()
        # add agents
        self.num = num

        # make initial conditions
        self.reset_world(world)
        self.calculate_pollution_map(world)
        return world
    
    def reset_world(self, world):
        world.timestep = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([2.5,0,0])
        world.agents[0].color = np.array([1.0,1.0,0.0])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.0,1.0,0.0])
        # walls     
        for i, wall in enumerate(world.walls):
            wall.color = np.array([0.70,0.05,0.405])

        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_pos = np.array([0.0,0.0])
            landmark.state.p_vel = np.zeros(world.dim_p)
            
        for i, wall in enumerate(world.walls):
            if i == 0:
                wall.state.p_pos = np.array([0.0,1.0])
                wall.state.p_vel = np.zeros(world.dim_p)
               
            if i == 1:
                wall.state.p_pos = np.array([0.0,-1.0])
                wall.state.p_vel = np.zeros(world.dim_p)
            if i == 2:
                wall.state.p_pos = np.array([-1.0,0.0])
                wall.state.p_vel = np.zeros(world.dim_p)
            if i == 3:
                wall.state.p_pos = np.array([1.0,0.0])
                wall.state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.v = 0.002
            agent.state.delta = np.random.uniform(0,360)
            agent.state.v_cmd = 0.0
            agent.state.delta_cmd = 0.0
            # agent.state.heading = np.random.uniform(0,2*np.pi)
            # agent.old_targetDis = self.get_distance(agent.state.p_pos, world.landmarks[0].state.p_pos)  
            

    def calculate_pollution_map(self, world):
        #the counting has been amplified by 1000 times
        xPos = int(world.landmarks[0].state.p_pos[0]/world.interval)
        yPos = int(world.landmarks[0].state.p_pos[1]/world.interval)

        for distFrmSrc in range(0, int(world.landmarks[0].size/(3*world.interval)+1)):
            for angle in np.arange(0,360,0.1):
                xPollu = distFrmSrc * np.math.cos((3.142/180) * angle)
                yPollu = distFrmSrc * np.math.sin((3.142/180) * angle)
				
                tempX = xPos + int(xPollu)
                tempY = yPos + int(yPollu)
                #transform coordinate
                tempX1 = tempX + int(world.xsize/2)
                tempY1 = tempY + int(world.ysize/2)
                world.pollution_map[tempX1][tempY1] = 3

        for distFrmSrc in range(int(world.landmarks[0].size/(3*world.interval)+1), int(world.landmarks[0].size/(1.5*world.interval)+1)):
            for angle in np.arange(0,360,0.1):
                xPollu = distFrmSrc * np.math.cos((3.142/180) * angle)
                yPollu = distFrmSrc * np.math.sin((3.142/180) * angle)
				
                tempX = xPos + int(xPollu)
                tempY = yPos + int(yPollu)

                #transform coordinate
                tempX1 = tempX + int(world.xsize/2)
                tempY1 = tempY + int(world.ysize/2)
                world.pollution_map[tempX1][tempY1] = 2 
        
        for distFrmSrc in range(int(world.landmarks[0].size/(1.5*world.interval)+1), int(world.landmarks[0].size/world.interval+1)):
            for angle in np.arange(0,360,0.01):
                xPollu = distFrmSrc * np.math.cos((3.142/180) * angle)
                yPollu = distFrmSrc * np.math.sin((3.142/180) * angle)
				
                tempX = xPos + int(xPollu)
                tempY = yPos + int(yPollu)

                #transform coordinate
                tempX1 = tempX + int(world.xsize/2)
                tempY1 = tempY + int(world.ysize/2)
                world.pollution_map[tempX1][tempY1] = 1

       
    def reward(self, d, n_agents, wall_coli, des):
        """
        Get rewards for each agent based on distances to other boids

        Args:
            d (np.array): 2d array representing euclidean distances between
                each pair of boids

        Returns:
            np.array: 1d array of reward values for each agent
        """
        proximity_threshold = 0.04
        distant_threshold = 0.1

        dis_rewards = _distance_rewards(
            d, proximity_threshold, distant_threshold,
        )

        wall_coli1 = np.any(wall_coli,axis=1)
        func = lambda x: -1000 if x==1 else 0 
        wall_penalties = [func(i) for i in wall_coli1]
        wall_penalties = np.array(wall_penalties)

        func1 = lambda x: 1000 if x<0.1 else 0
        goal_rewards =  [func1(i) for i in des]
        goal_rewards = np.array(goal_rewards)


        return dis_rewards + wall_penalties + goal_rewards
    
    def done(self, world, t_max):
        # if target_d < 0.1 or np.any(agent.wall_collision) or world.timestep == t_max:
        # if np.any(agent.wall_collision) or world.timestep == t_max:
        if world.timestep == t_max:
           return 1
        else:
           return 0

    def get_distance(self,p1,p2):
        return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def get_bearing(self,p1,p2):
        ang = math.atan2(p2[0]-p1[0], p2[1]-p1[1])

        if ang<=-np.pi/2:
           return ang + 3*np.pi/2
        else:
           return ang - np.pi/2
        
    def get_turning_angle(self,a,b):
        a1 = np.sqrt((a[0] * a[0]) + (a[1] * a[1]))
        b1 = np.sqrt((b[0] * b[0]) + (b[1] * b[1]))
        aXb = (a[0] * b[0]) + (a[1] * b[1])

        cos_ab = aXb/(a1*b1)
        angle_ab = math.acos(cos_ab)*(180.0/np.pi)
        return angle_ab

    def point_direction(self,a,b,c):
        #start, end , point S = (x1-x3)*(y2-y3)-(y1-y3)*(x2-x3) 
        #S>0, left; S<0, right; S=0, on the line
        S = (a[0]-c[0])*(b[1]-c[1])-(a[1]-c[1])*(b[0]-c[0])
        if S < 0:
            return 1
        if S > 0:
            return -1
        if S == 0:
            return 1

    def agent_turning_angle(self, agent, world):
        a = world.landmarks[0].state.p_pos - agent.state.p_pos
        b = np.array([math.cos(np.radians(agent.state.delta)),math.sin(np.radians(agent.state.delta))])
        angle = self.get_turning_angle(a,b)
        a1 = agent.state.p_pos
        b1 = np.array([math.cos(np.radians(agent.state.delta)),math.sin(np.radians(agent.state.delta))])
        c1 = world.landmarks[0].state.p_pos

        return angle*self.point_direction(a1,b1,c1)


    def observation(self, x, xd, theta, n_agents, n_nearest) -> np.array:
        """
        theta is radians
        Returns a view on the flock phase space local to each agent. Since
        in this case all the agents move at the same speed we return the
        x and y components of vectors relative to each boid and the relative
        heading relative to each agent.

        In order for the agents to have similar observed states, for each agent
        neighbouring boids are sorted in distance order and then the closest
        neighbours included in the observation space

        Returns:
            np.array: Array of local observations for each agent, bounded to
                the range [-1,1]
        """
        xs = _product_difference(x[0], n_agents)
        ys = _product_difference(x[1], n_agents)
        d = _distances(xs, ys)
        
        # Sorted indices of flock members by distance
        sort_idx = np.argsort(d, axis=1)[:, : n_nearest]

        #print(self.theta.shape)
        relative_headings = _relative_headings(theta)
        # print(relative_headings)

        closest_x = take_along_axis(xs, sort_idx) #np.sort(a,axis=1)
        closest_y = take_along_axis(ys, sort_idx)
        closest_h = take_along_axis(relative_headings, sort_idx)
        # print(closest_y)
        # Rotate relative co-ords relative to each boids heading
        cos_t = np.cos(theta)[:, np.newaxis]
        sin_t = np.sin(theta)[:, np.newaxis]
        
        x1 = (cos_t * closest_x + sin_t * closest_y) / 2.8
        y1 = (cos_t * closest_y - sin_t * closest_x) / 2.8

        des_xs = (xd[0] - x[0])[:,np.newaxis]
        des_ys = (xd[1] - x[1])[:,np.newaxis]

        des_x = (cos_t * des_xs + sin_t * des_ys) / 1.4
        des_y = (cos_t * des_ys - sin_t * des_xs) / 1.4

        #walls obstacle
        wall_dists = []
        wall_angles = []
        for i in range(n_agents):
            wall_dists.append(np.array([np.abs(1.0-x[1,i]),np.abs(1.0+x[1,i]),np.abs(1.0+x[0,i]),np.abs(1.0-x[0,i])])/2) # rangefinder: forward, back, left, right
            wall_angles.append(np.array([np.pi / 2, 3 / 2 * np.pi, np.pi, 0.0]) - theta[i])
        #print(len(entity_dis),len(wall_dists))
        wall_dists = np.r_[wall_dists]
        wall_angles = np.r_[wall_angles]

        wall_obs = np.zeros([n_agents,3])
        closest_wall = np.argmin(wall_dists,axis=1)

        for j in range(n_agents):
            wall_obs[j,0] = wall_dists[j,closest_wall[j]]
            wall_obs[j,1] = np.cos(wall_angles[j,closest_wall[j]])
            wall_obs[j,2] = np.sin(wall_angles[j,closest_wall[j]])

    
        local_observation = np.concatenate(
            [x1, y1, closest_h, des_x, des_y, wall_obs], axis=1
        )
        
        return d, local_observation