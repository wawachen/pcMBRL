from os import path
from os.path import dirname, join, abspath

from numpy.core.defchararray import add
from pyrep import PyRep
from pyrep.envs.drone_RL_agent import Drone

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np

from cv2 import VideoWriter, VideoWriter_fourcc
import cv2
from pyrep.objects.vision_sensor import VisionSensor
import random
import math

#---------for optical flow image---------------
import sys
sys.path.append('/home/wawa/RL_transport_3D/core')
import glob
from PIL import Image

from raft import RAFT
from utils.utils import InputPadder
import argparse

#----------------------------------------------

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import math

import torch
from pyrep.policies.utilities1 import soft_update, transpose_to_tensor, transpose_list, hard_update
from pyrep.policies.utilities1 import _relative_headings, _shortest_vec, _distances, _relative_headings, _product_difference, _distance_rewards, take_along_axis

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Drone_Env:

    def __init__(self,env_name,num_agents):
        self.reset_callback = self.reset_world
        self.reward_callback = self.reward_and_terminate
        self.observation_callback = self.observation
        # self.done_callback = self.done

        # environment parameters
        self.discrete_action_space = False
        self.time_step = 0
        self.prev_im = np.zeros([100,512,3])
        # self.prev_im = np.zeros([60,32,3])

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.num_a = num_agents
        self.env_name = env_name
        self.close_simulation = False

        self.safe_distance = 0.71
        self.x_limit_min = -7.5+self.safe_distance/2
        self.x_limit_max = 7.5-self.safe_distance/2
        self.y_limit_min = -7.5+self.safe_distance/2
        self.y_limit_max = 7.5-self.safe_distance/2

        self.shared_reward = True

        #--------optical flow-----------------
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        self.opt_args = parser.parse_args()
        self.opt_args.model = '/home/wawa/RL_transport_3D/pyrep/networks/raft-things.pth'

        self.model = torch.nn.DataParallel(RAFT(self.opt_args))
        self.model.load_state_dict(torch.load(self.opt_args.model))

        self.model = self.model.module
        self.model.to(device)
        self.model.eval()
        #---------------------------------------

        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles = self.import_agent_models()
        self.agents = [Drone(i) for i in range(num_agents)]
        self.goals = self.generate_goal()

        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(5) #3*3
            else:
                u_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

            total_action_space.append(u_action_space)
            self.action_space.append(total_action_space[0])
            #observation space
            # obs_dim = len(self.observation_callback(agent))
            # # print(obs_dim)
            # self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            # agent.action.c = np.zeros(self.world.dim_c)

    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL.ttm'))
            model_handles.append(m1)

        # [m,m1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_1,m1_1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_2,m1_2]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_3,m1_3]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_4,m1_4]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        # [m_5,m1_5]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))

        # model_handles = [m1,m1_1,m1_2,m1_3,m1_4,m1_5]
        return model_handles

    
    def check_collision_a(self,agent1,agent2):
        delta_pos = agent1.agent.get_drone_position()[:2] - agent2.agent.get_drone_position()[:2]
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        return True if dist <= self.safe_distance else False

    def check_collision_p(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance < 1.0:
            return 1
        else:
            return 0

    def generate_goal(self):
        #visualization goal
        # targets = [Shape.create(type=PrimitiveShape.CYLINDER,
        #               size=[0.36, 0.36, 0.01],
        #               color=[1.0, 0.1, 0.1],
        #               static=True, respondable=False) for i in range(self.num_a)]
        # targets = [Shape('goal'), Shape('goal0'), Shape('goal1')]
        goal_points = np.array([[0,0,1.7]]) #[-1.25,0,0.005],[0,1.25,0.005],[1.25,0,0.005]

        # for i in range(self.num_a):
        #     targets[i].set_position(goal_points[i])

        return goal_points

    def random_spread_without_obstacle(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        for i in range(self.num_a):
            if i == 0:
                [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL.ttm'))
                model_handles.append(m1)
                objs.append(m)
                self.agents.append(Drone(0))
                r = np.random.uniform(3.0,6.0)
                angle = np.random.uniform(0,360)
                angle = np.radians(angle)
                vpt = [r*np.cos(angle),r*np.sin(angle)]
                self.agents[0].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
            else:
                [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL.ttm'))
                model_handles.append(m1)
                objs.append(m)
                self.agents.append(Drone(i))
                r = 1.5
                angle = 90*(i-1)
                angle = np.radians(angle)
                # print(angle)
                vpt = [r*np.cos(angle),r*np.sin(angle)]
                # print(vpt)
                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])


            # self.agents[i].get_agent((vpt[0],vpt[1]))
            # print("all",vpts)
            # print("current",vpt)
            
        return model_handles,objs

   
    def reset_world(self):
        self.time_step = 0
        
        #self.suction_cup.release()
        if not self.close_simulation:
            for i in range(self.num_a):
                self.pr.remove_model(self.model_handles[i])

        # self.model_handles = self.import_agent_models()
        # # self.agents = [Drone(i) for i in range(self.num_a)]
        # self.random_position_spread()

        self.model_handles,ii = self.random_spread_without_obstacle()
        if self.close_simulation:
            self.goals = self.generate_goal()

        for j in range(self.num_a):
            self.agents[j]._reset()
    
        #for hovering when reset
        for j in range(50):
            for agent in self.agents:
                agent.hover(1.7)
            self.pr.step()
     
        obs_n = []
        goal = []
        
        obs_n.append(self.observation_callback(self.agents[0])[1])
        goal.append(self.observation_callback(self.agents[0])[0])
        self.close_simulation = False

        return obs_n,goal


    def step_orca(self, action_n):
        positions_n = []
        velocities_n = []
        planes_n = []
        prefer_vels = np.zeros([self.num_a,2])

        added_agents = []
        added_nodes = []

        for i in range(self.goals.shape[0]):  # world.entities:
            goal_pos = np.zeros(self.num_a)
            for j, agent in enumerate(self.agents):
               if j in added_agents:
                  goal_pos[j] = 100
               else:
                  goal_pos[j] = (np.sqrt(np.sum((self.goals[i,:2]-agent.get_2d_pos())**2))) 
            num_min = np.argmin(goal_pos)
            look_neighbour_dis = np.zeros(self.goals.shape[0])
            #minimum agent  look up all goals
            look_neighbour_dis[i] = goal_pos[num_min]

            for m in range(self.goals.shape[0]):
                if m == i: continue
                if m in added_nodes:
                    look_neighbour_dis[m] = 100
                else:
                    look_neighbour_dis[m] = np.sqrt(np.sum((self.goals[m,:2]-self.agents[num_min].get_2d_pos())**2))

            while np.argmin(look_neighbour_dis)!=i:
                goal_pos[num_min] = 100
                num_min = np.argmin(goal_pos)

                look_neighbour_dis = np.zeros(self.goals.shape[0])
                #minimum agent  look up all goals
                look_neighbour_dis[i] = goal_pos[num_min]
                for m in range(self.goals.shape[0]):
                    if m == i: continue
                    if m in added_nodes:
                        look_neighbour_dis[m] = 100
                    else:
                        look_neighbour_dis[m] = np.sqrt(np.sum((self.goals[m,:2]-self.agents[num_min].get_2d_pos())**2))

            prefer_vel = self.goals[i,:2]-self.agents[num_min].get_2d_pos()
            vel_len = np.sqrt(prefer_vel[0]**2+prefer_vel[1]**2)
            if vel_len>1: #max speed 1
                prefer_vel = (prefer_vel/vel_len)*1.0
            prefer_vels[num_min,0] = prefer_vel[0]
            prefer_vels[num_min,1] = prefer_vel[1]
            added_agents.append(num_min)
            added_nodes.append(i)

        for i, agent in enumerate(self.agents):
            for other in self.agents:
                if other is agent:
                    continue
                positions_n.append(np.array([other.get_2d_pos()[0],other.get_2d_pos()[1]]))
                velocities_n.append(np.array([other.get_2d_vel()[0],other.get_2d_vel()[1]]))
            
            agent.update_agent()
            # agent.agent.set_prefer_velocity()
            # print(positions_n)
            agent.computeNeighbors(positions_n,velocities_n)
            agent.computeNewVelocity()
            planes = agent.get_planes()
            planes_n.append(planes)

            positions_n = []
            velocities_n = []

        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        for j in range(10):
            # set action for each agent
            for i, agent in enumerate(self.agents):
                agent.set_action(action_n[i])
                # agent.set_action_pos(action_n[i], self.pr)
            self.pr.step()

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            #----------------------------
            obs_n.append(self.observation_callback(agent))
            rw,ter = self.reward_callback(agent)
            reward_n.append(rw)
            done_n.append(ter)

        #all agents get total reward in cooperative case
        reward = np.sum(reward_n[:]) #need modify
        if self.shared_reward:
            reward_n = [reward] * self.num_a

        self.time_step+=1

        return np.array(obs_n), np.array(reward_n), np.array(done_n), planes_n, prefer_vels

    def step(self, action_n):

        obs_n = []
        reward_n = []
        done_n = []
        goal_n = []

        for i, agent in enumerate(self.agents):
            p_pos = agent.agent.get_drone_position()
            agent.agent.panoramic_holder.set_position([p_pos[0],p_pos[1],p_pos[2]+0.1],reset_dynamics=False)
            agent.agent.panoramic_holder.set_orientation([0,0,np.radians(90)],reset_dynamics=False)

        for j in range(10):
            # set action for each agent
            for i, agent in enumerate(self.agents):                    
                agent.set_action(action_n[i])
                # agent.set_action_pos(action_n[i], self.pr)
            self.pr.step()

        # record observation for each agent
        
        #----------------------------
        obs_n.append(self.observation_callback(self.agents[0])[1])
        goal_n.append(self.observation_callback(self.agents[0])[0])
        rw,ter = self.reward_callback(self.agents[0])
        reward_n.append(rw)
        done_n.append(ter)

        #all agents get total reward in cooperative case
        # reward = np.sum(reward_n[:]) #need modify
        # if self.shared_reward:
        #     reward_n = [reward] * self.num_a

        self.time_step+=1

        return np.array(obs_n), np.array(goal_n), np.array(reward_n), np.array(done_n)


    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def restart(self):
        if self.pr.running:
            self.pr.stop()
        self.pr.shutdown()

        self.pr = PyRep()
        self.pr.launch(self.env_name, headless=False)
        self.pr.start()
        self.close_simulation = True
        

    def reward_and_terminate(self, agent):
        rew = 0
        terminate = 0
        finish_sig = 0

        dists = [np.sqrt(np.sum(np.square(agent.get_2d_pos() - self.goals[0,:2])))]
        finish_sig = np.any((np.array(dists)<0.35))
        rew -= dists[0]
        
        if finish_sig:
            terminate = 1
            rew = 0
            rew += 1

        if agent.agent.assess_collision():
            rew-=1
            terminate = 1
        
        if agent.agent.get_position()[2]<1.3:
            terminate = 1

        return rew,terminate


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

    # def agent_turning_angle(self, agent, world):
    #     a = world.landmarks[0].state.p_pos - agent.state.p_pos
    #     b = np.array([math.cos(np.radians(agent.state.delta)),math.sin(np.radians(agent.state.delta))])
    #     angle = self.get_turning_angle(a,b)
    #     a1 = agent.state.p_pos
    #     b1 = np.array([math.cos(np.radians(agent.state.delta)),math.sin(np.radians(agent.state.delta))])
    #     c1 = world.landmarks[0].state.p_pos

    #     return angle*self.point_direction(a1,b1,c1)


    def observation(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i in range(self.goals.shape[0]):  # world.entities:
            entity_pos.append((self.goals[i,:2]-agent.get_2d_pos())/15)   
        # entity_pos.append(agent.target-agent.state.p_pos)
        
        image = agent.get_rgb_image()

        #-------------
        with torch.no_grad():
            # print(self.prev_im.shape)
            self.prev_im = torch.from_numpy(self.prev_im).permute(2, 0, 1).float()
            image1 = self.prev_im[None].to(device)

            # print(image.shape)
            image2_1 = torch.from_numpy(image).permute(2, 0, 1).float()
            image2 = image2_1[None].to(device)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)

        #-------------
        # print(flow_up.max())
        # pos_obs = agent.agent.get_position()[:2]/2.5
        self.prev_im = image

        return (np.concatenate(entity_pos),flow_up[0].cpu().numpy()/255.0) # output [channel,h,w]



