from os import path
from pyrep import PyRep
from pyrep.envs.drone_RL_agent import Drone_s

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
import random

import gym
from gym import spaces
import numpy as np


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Drone_Env:

    def __init__(self,env_name,num_agents,num_goals):
        self.reset_callback = self.reset_world
        self.reward_callback = self.reward_and_terminate
        self.observation_callback = self.observation
        # self.done_callback = self.done

        # environment parameters
        self.discrete_action_space = False
        self.time_step = 0
        self.sight = 5.0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.num_a = num_agents
        self.num_p = num_goals
        self.env_name = env_name
        self.close_simulation = False

        self.safe_distance = 0.71
        self.x_limit_min = -7.5+self.safe_distance/2
        self.x_limit_max = 7.5-self.safe_distance/2
        self.y_limit_min = -7.5+self.safe_distance/2
        self.y_limit_max = 7.5-self.safe_distance/2

        self.shared_reward = True
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles = self.import_agent_models()
        self.agents = [Drone_s(i) for i in range(num_agents)]
        self.payload = Shape('Cuboid4')
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
            obs_dim = len(self.observation_callback(agent))
            # print(obs_dim)
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            # agent.action.c = np.zeros(self.world.dim_c)

    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"   
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)

        return model_handles


    def generate_goal(self):
        #visualization goal
        targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.56, 0.56, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_p)]
        # targets = [Shape('goal'), Shape('goal0'), Shape('goal1')]
        if self.num_a == 6:
            # goal_points = np.array([[-2.8,-2.0,1.7],[-2.8,0.0,1.7],[2.8,-2.0,1.7],[2.8,0.0,1.7],[0.0,0.0,1.7],[0.0,-2.0,1.7]])
            # goal_points = np.array([[2.8,0.0,1.7],[0.0,0.0,1.7],[-2.8,0.0,1.7],[2.8,-2.0,1.7],[0.0,-2.0,1.7],[-2.8,-2.0,1.7]]) #[-1.25,0,0.005],[0,1.25,0.005],[1.25,0,0.005]
            goal_points = np.array([[2.25,-2.25,1.7],[-2.25,-2.25,1.7],[1.2,0,1.7],[-1.2,0,1.7],[2.25,2.25,1.7],[-2.25,2.25,1.7]]) #[-1.25,0,0.005],[0,1.25,0.005],[1.25,0,0.005]
        if self.num_a == 3:
            goal_points = np.array([[2.25,-2.25,1.7],[-2.25,-2.25,1.7],[1.2,0,1.7]]) #[-1.25,0,0.005],[0,1.25,0.005],[1.25,0,0.005]
           
        if self.num_a == 12:
            goal_points = np.array([[3.6,-3.6,1.7],[0,-3.6,1.7],[-3.6,-3.6,1.7],[1.8,-1.8,1.7],[-1.8,-1.8,1.7],[3.6,0,1.7],[-3.6,0,1.7],[1.8,1.8,1.7],[-1.8,1.8,1.7],[3.6,3.6,1.7],[0,3.6,1.7],[-3.6,3.6,1.7]])
        
        for i in range(goal_points.shape[0]):
            targets[i].set_position(goal_points[i])

        return goal_points

    def random_spread_without_obstacle(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Drone_s(i))
            
            vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
            self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
            check_conditions = self.agents[i].agent.assess_collision()

            while check_conditions:
                vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
                check_conditions = self.agents[i].agent.assess_collision()

        return model_handles,objs

    def direct_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"

        # pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
        pos = np.array([[2.5,3.5],[0-2.5,3.5],[2.5,-3.5],[-2.5,-3.5],[4.5,0],[-4.5,0]])

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Drone_s(i))
            
            self.agents[i].agent.set_3d_pose([pos[i,0],pos[i,1],1.7,0.0,0.0,0.0])
                

        return model_handles,objs
    

    def reset_world(self):
        self.time_step = 0
        
        #self.suction_cup.release()
        if not self.close_simulation:
            for i in range(self.num_a):
                self.pr.remove_model(self.model_handles[i])

        self.payload.set_position([0.0,0.0,0.3])
        if self.num_a==3:
            self.payload.set_orientation([0.0,np.radians(-90),0.0])
        else:
            self.payload.set_orientation([np.radians(0),0.0,np.radians(0)])

        # self.model_handles,ii = self.direct_spread()
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
       
        for agent in self.agents:
            obs_n.append(self.observation_callback(agent))
         
        self.close_simulation = False

        return np.array(obs_n)


    def step(self, action_n):

        obs_n = []
        reward_n = []
        done_n = []

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

        return np.array(obs_n), np.array(reward_n), np.array(done_n)


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
        finish_sig = np.zeros(self.goals.shape[0])

        #team reward
        for i in range(self.goals.shape[0]):
            dists = [np.sqrt(np.sum(np.square(a.get_2d_pos() - self.goals[i,:2]))) for a in self.agents]
            finish_sig[i] = np.any((np.array(dists)<0.35))
            rew -= min(dists)/15
            
        
        if np.all(finish_sig):
            # terminate = 1
            rew = 0
            rew += 1
        
        #individual reward
        if agent.agent.assess_collision():
            rew-=1
            terminate = 1
       
        if agent.agent.get_position()[2]<1.3:
            terminate = 1

        return rew,terminate


    def observation(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
    
        for i in range(self.goals.shape[0]):  # world.entities:
            entity_pos.append((self.goals[i,:2]-agent.get_2d_pos())/15)   
        # entity_pos.append(agent.target-agent.state.p_pos)
        
        # communication of all other agents
        other_pos = []
        for other in self.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append((other.get_2d_pos() - agent.get_2d_pos())/15.0)

        other_vel = []
        for other in self.agents:
            if other is agent: continue
            other_vel.append(other.get_2d_vel())

        # pos_obs = agent.agent.get_position()[:2]/2.5

        return np.concatenate([agent.get_2d_vel()]+ [agent.get_2d_pos()/15] + other_vel + other_pos + entity_pos)


    #local observation 
    def local_observation(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i in range(self.goals.shape[0]):  # world.entities:
            distance = np.sqrt(np.sum(np.square(self.goals[i,:2]-agent.get_2d_pos())))
            if distance>self.sight:
                entity_pos.append([0,0,0])
            else:
                entity_pos.append((self.goals[i,:2]-agent.get_2d_pos())/15) 
                entity_pos.append([1])  
        # entity_pos.append(agent.target-agent.state.p_pos)
        
        # communication of all other agents
        other_live = []
        other_pos = []
        other_vel = []
        for other in self.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            distance = np.sqrt(np.sum(np.square(other.get_2d_pos()-agent.get_2d_pos())))
            if distance>self.sight:
                other_pos.append([0,0])
                other_vel.append([0,0])
                other_live.append([0])
            else:
                other_pos.append((other.get_2d_pos() - agent.get_2d_pos())/15.0)
                other_vel.append(other.get_2d_vel())
                other_live.append([1])

        # pos_obs = agent.agent.get_position()[:2]/2.5

        return np.concatenate([agent.get_2d_vel()]+ [agent.get_2d_pos()/15] + other_vel + other_pos + other_live + entity_pos)



