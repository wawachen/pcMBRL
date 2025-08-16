from os import path
from pyrep import PyRep
from pyrep.envs.turtle_RL_agent import Turtle_o

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

import random
from gym import spaces
import numpy as np
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Turtle_Env:

    def __init__(self, args, env_name,num_agents):
        self.args = args
        self.reset_callback = self.reset_world
        self.reward_callback = self.reward_and_teriminate
        self.observation_callback = self.observation
        # self.done_callback = self.done

        # environment parameters
        self.discrete_action_space = False
        self.time_step = 0

        self.field_size = args.field_size/2

        self.env_name = env_name
        self.close_simulation = False

        # configure spaces
        self.x_limit_min = -5+0.2
        self.x_limit_max = 5-0.2
        self.y_limit_min = -5+0.2
        self.y_limit_max = 5-0.2
        self.safe_distance = 0.3545

        self.action_space = []
        self.observation_space = []
        self.num_a = num_agents
        
        self.shared_reward = True
        self.sight_range = 2.0
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
       
        self.model_handles,ii = self.import_agent_models()
       
        self.agents = [Turtle_o(i) for i in range(num_agents)] #because it only import odd index 
        self.targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.56, 0.56, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_a)]
        self.goals = self.generate_goal()

         # physical action space
        # 第一个动作范围在[-1,1]，第二个动作范围在[0,1]
        low = np.array([-1, -1] * num_agents, dtype=np.float32)
        high = np.array([1, 1] * num_agents, dtype=np.float32)
        u_action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = u_action_space
        #observation space
        obs_dim = len(self.observation_callback())
        # print(obs_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)

    def import_agent_models(self):
        robot_DIR = "/home/xlab/MARL_transport/examples/models"
        
        #pr.remove_model(m1)
        model_handles = []
        objs = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'turtlebot_beta.ttm'))
            model_handles.append(m1)
            objs.append(m)

        return model_handles,objs

    def generate_goal(self):
        #visualization goal 
        if self.num_a == 3:
            goals = np.array([[3.0, 0], [-3.0, 0], [0, 0]])
        elif self.num_a == 4:
            goals = np.array([[2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]])
        elif self.num_a == 6:
            goals = np.array([[3.0, -3.0], [-3.0, -3.0], [1.5, 0], [-1.5, 0], [3.0, 3.0], [-3.0, 3.0]])

        points = []

        for i in range(len(goals)):
            points.append([goals[i][0],goals[i][1],0.5])
            
        goal_points = np.array(points)

        for i in range(self.num_a):
            self.targets[i].set_position(goal_points[i])

        return goal_points


    def check_collision(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance < 1.0:
            return 1
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.agent.get_position()[:2] - agent2.agent.get_position()[:2]
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return True if dist < self.safe_distance else False

    def random_position_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/xlab/MARL_transport/examples/models"
        saved_agents = []
        vpts = []
        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'turtlebot_beta.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Turtle_o(i))

            self.agents[i].agent.set_motor_locked_at_zero_velocity(True)
            if i == 0:
                self.agents[i].agent.set_3d_pose([random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max), 0.0607,0.0,0.0,np.radians(random.uniform(-180,180))])
                vx = self.agents[i].agent.get_position()[0]
                vy = self.agents[i].agent.get_position()[1]
                vpts.append([vx,vy])
                saved_agents.append(i)
            else:
                vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
                check_conditions = np.sum(check_list)

                while check_conditions:
                    vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                    check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
                    check_conditions = np.sum(check_list)

                # print("all",vpts)
                # print("current",vpt)
                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],0.0607,0.0,0.0,np.radians(random.uniform(-180,180))])
                vpts.append(vpt)
                saved_agents.append(i)
        # print(vpts)
        return model_handles,objs

    def direct_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/xlab/MARL_transport/examples/models"

        if self.num_a == 3:
            pos = np.array([[2.0,2.5],[-2.5,2.5],[0,-2.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)
        
        if self.num_a == 4:
            pos = np.array([[3.5,4.5],[3.5,-4.5],[-3.5,-4.5],[-3.5,4.5]])
            # pos = np.array([[2.5,4.5],[3.5,4.5],[-2.5,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        if self.num_a == 6:
            pos = np.array([[4.5,4.5],[4.5,-4.5],[-4.5,4.5],[-4.5,-4.5],[0,4.5],[0,-4.5]])
            # pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        for i in range(self.num_a):
            # [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            # model_handles.append(m1)
            # objs.append(m)
            # self.agents.append(Drone_s(i))
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'turtlebot_hybrid.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Turtle_o(i))

            self.agents[i].agent.set_motor_locked_at_zero_velocity(True)
            #0.0607
            self.agents[i].agent.set_3d_pose([pos[i,0],pos[i,1],0.0607,0.0,0.0,0.0])
                

        return model_handles,objs


    def reset_world(self):
        #self.suction_cup.release()
        if not self.close_simulation:
            for i in range(self.num_a):
                self.pr.remove_model(self.model_handles[i])
      
        # self.model_handles,ii = self.import_agent_models()
        # start_index = int(ii[0].get_name().split("#")[1])
        # print(start_index)
        # self.agents = [Turtle(i) for i in range(self.num_a)]
        self.model_handles,ii = self.random_position_spread()
        if self.close_simulation:
            self.goals = self.generate_goal()

        self.time_step += 1

        obs_n = self.observation_callback()

        self.close_simulation = False

        return np.array(obs_n)
    
    def restart(self):
        if self.pr.running:
            self.pr.stop()
        self.pr.shutdown()

        self.pr = PyRep()
        self.pr.launch(self.env_name, headless=False)
        self.pr.start()
        self.close_simulation = True

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        done_ter = []
        
        for j in range(10):
            # set action for each agent
            for i, agent in enumerate(self.agents):
                agent.set_action(action_n[2*i:2*i+2])
            self.pr.step()

        obs_n = self.observation_callback()

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            rw,ter,dter = self.reward_callback(agent)
            reward_n.append(rw)
            done_n.append(ter)
            done_ter.append(dter)

        #all agents get total reward in cooperative case
        reward = reward_n[0] #need modify

        if np.all(done_ter):
            reward = 1*self.num_a

        #once collision every agent will be pulished
        if np.any(done_n):
            #reward = -50
            reward = reward - 1*self.num_a

        done_all = np.any(done_n)

        if self.shared_reward:
            reward_n = [reward] * self.num_a

        self.time_step+=1

        return np.array(obs_n), np.array(reward_n), done_all, np.all(done_ter)

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def reward_and_teriminate(self, agent):
       # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        done_terminate = 0
        terminate = 0
        finish_sig = np.zeros(self.num_a)

        for i in range(self.goals.shape[0]):
            dists = [np.sqrt(np.sum(np.square(a.agent.get_position()[:2] - self.goals[i,:2]))) for a in self.agents]
            finish_sig[i] = np.any((np.array(dists)<0.3))
            rew -= min(dists)/(self.field_size*2)
        
        if np.all(finish_sig):
            done_terminate = 1

        # collision detection
        wall_dists = np.array([np.abs(5-agent.agent.get_position()[1]),np.abs(5+agent.agent.get_position()[1]),np.abs(5+agent.agent.get_position()[0]),np.abs(5-agent.agent.get_position()[0])]) # rangefinder: forward, back, left, right
        wall_sig = np.any(wall_dists<0.206)

        agent_collision = []
        for a in self.agents:
            if a == agent: continue
            if self.is_collision(agent,a):
                agent_collision.append(1)
            else:
                agent_collision.append(0)
        agent_sig = np.any(np.array(agent_collision))

        if wall_sig or agent_sig:
            terminate = 1
        
        return rew,terminate,done_terminate

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

    def get_local_goal(self,agent, goals):
        pos = agent.agent.get_position()[:2]
        orientation = agent.agent.get_orientation()[2]
        x = pos[0]
        y = pos[1]
        theta = orientation

        goal_x = goals[0]
        goal_y = goals[1]

        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return np.array([local_x, local_y])

    def get_local_goal1(self, position, heading, goals):
        pos = position
        orientation = heading
        x = pos[0]
        y = pos[1]
        theta = orientation

        goal_x = goals[0]
        goal_y = goals[1]

        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def get_local_goal2(self, positions, headings, goals):
        """
        并行计算多个智能体的局部目标坐标。
        positions: (N, 2) array，每行是一个智能体的位置 [x, y]
        headings: (N,1) array，每个智能体的朝向（弧度）
        goals: (N, 2) array，每行是对应目标点的位置 [goal_x, goal_y]
        返回: (N, 2) array，每行是局部目标坐标 [local_x, local_y]
        """
        x = positions[:, 0]
        y = positions[:, 1]
        theta = headings

        goal_x = goals[:, 0]
        goal_y = goals[:, 1]

        dx = goal_x - x
        dy = goal_y - y

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        local_x = dx * cos_theta[:,0] + dy * sin_theta[:,0]
        local_y = -dx * sin_theta[:,0] + dy * cos_theta[:,0]

        return np.concatenate([local_x.reshape(-1,1), local_y.reshape(-1,1)], axis=1)

    def observation(self):
        """
        Returns the observation for the given agent, organized in the same way as orca_bacterium_environment_demo_v0_MBRL.
        For each agent, the observation includes:
        - agent's own state (4): [vx, vy, local_goal_x, local_goal_y]
        - for each other agent:
            - relative position (2): [rel_x, rel_y]
            - relative velocity (2): [rel_vx, rel_vy]
        """
        # 为每个智能体构建观测
        observations = []
        
        for i, agent in enumerate(self.agents):
            obs = []
            
            # 1. 自身状态 (7维)
            self_pos = agent.get_2d_pos() / self.field_size
            self_vel = agent.get_2d_local_vel()
            self_ori = agent.get_orientation()
            
            obs.append(self_pos)  # 2维: [x, y]
            obs.append(self_vel)  # 2维: [vx, vy]
            obs.append(np.array([np.sin(self_ori[2]), np.cos(self_ori[2])])) # 2维: [roll, pitch， yaw]
            
            # 2. 其他智能体相对状态 (4维 × (n-1)个智能体)
            for j, other_agent in enumerate(self.agents):
                if j == i:  # 跳过自己
                    continue
                    
                # other_pos = self.get_local_goal(agent, other_agent.get_2d_pos())/ self.field_size
                other_pos = other_agent.get_2d_pos() / self.field_size
                other_ori = other_agent.get_orientation()
                
                obs.append(self_pos-other_pos)     # 2维: [rel_x, rel_y]
                obs.append(np.array([np.sin(self_ori[2]-other_ori[2]),np.cos(self_ori[2]-other_ori[2])]))     # 2维: [rel_vx, rel_vy]
                # 2维: [rel_vx, rel_vy]
                # obs.append(distance)    # 1维: 距离
                # obs.append(bearing)     # 1维: 相对角度
            
            observations.append(np.concatenate(obs))
        
        return np.concatenate(observations)  # 返回


    




