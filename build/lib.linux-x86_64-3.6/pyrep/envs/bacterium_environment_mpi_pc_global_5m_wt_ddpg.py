from os import path
from pyrep import PyRep
from pyrep.envs.drone_RL_agent import Drone_s1

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
import random
from scipy.optimize import linear_sum_assignment

import gym
from gym import spaces
import numpy as np
import math
import random


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Drone_Env:

    def __init__(self,args, env_name):
        self.args = args
        self.reset_callback = self.reset_world
        self.reward_callback = self.reward_and_terminate
        # self.observation_callback = self.observation
        # self.done_callback = self.done

        # environment parameters
        self.discrete_action_space = False
        self.time_step = 0
        self.field_size = self.args.field_size / 2  #5mx5m

        self.is_local_obs = self.args.is_local_obs

        if self.is_local_obs:
            print("use local sight")
            self.sight = args.local_sight
            self.observation_callback = self.local_observation
        else:
            self.observation_callback = self.observation

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.num_a = self.args.n_agents
        self.env_name = env_name
        self.close_simulation = False

        self.safe_distance = 0.59 #0.5
        self.x_limit_min = -self.field_size+self.safe_distance/2+0.5
        self.x_limit_max = self.field_size-self.safe_distance/2-0.5
        self.y_limit_min = -self.field_size+self.safe_distance/2+0.5
        self.y_limit_max = self.field_size-self.safe_distance/2-0.5

        self.shared_reward = True
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles, self.load_handle = self.import_agent_models()
        self.agents = [Drone_s1(i) for i in range(self.args.n_agents)]

        self.payload = Shape('Cuboid4')
        self.targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.56, 0.56, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_a)]

        self.goals = self.generate_goal()

        # each column represents distance of all agents from the respective landmark
        world_dists = np.array([[np.linalg.norm(a.get_2d_pos() - self.goals[l,:2]) for l in range(self.goals.shape[0])]
                               	for a in self.agents])
        # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
        _, self.ci = linear_sum_assignment(world_dists)

        for i, agent in enumerate(self.agents):
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(5) #3*3
            else:
                u_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

            total_action_space.append(u_action_space)
            self.action_space.append(total_action_space[0])
            #observation space
            obs_dim = len(self.observation_callback(i,agent))
            # print(obs_dim)
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            # agent.action.c = np.zeros(self.world.dim_c)

    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"   
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state1.ttm'))
            model_handles.append(m1)

        if self.args.load_type == "three":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_three.ttm'))

        if self.args.load_type=="four":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_four.ttm'))
        
        if self.args.load_type=="six":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_six.ttm'))
        
        return model_handles,m2

    
    def check_collision_a(self,agent1,agent2):
        delta_pos = agent1.agent.get_drone_position()[:2] - agent2.agent.get_drone_position()[:2]
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        return True if (dist <= self.safe_distance) else False
    
    def check_collision_a_m(self,agent1,agent2):
        delta_pos = agent1.agent.get_drone_position()[:2] - agent2.agent.get_drone_position()[:2]
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        return True if dist <= 0.5 else False

    def check_collision_a_dist(self,agent1,agent2):
        delta_pos = agent1.agent.get_drone_position()[:2] - agent2.agent.get_drone_position()[:2]
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        return dist 

    def check_collision_p(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance < 2.0:
            return 1
        else:
            return 0

    def load_spread(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"  
        ######################################
        if self.args.load_type == "three":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_three.ttm'))

        if self.args.load_type == "four":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_four.ttm'))
        
        if self.args.load_type == "six":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_six.ttm'))
        #####################################
        self.payload = Shape('Cuboid4')
        self.payload.set_orientation([0.0,0.0,0.0])
        self.payload.set_position([0.0,0.0,0.1])

        return m2

    def generate_goal(self):
        #####################################
        if self.args.load_type == "three":
            self.payload_1 = Shape('Cuboid28')
            self.payload_2 = Shape('Cuboid29')
            self.payload_3 = Shape('Cuboid30')

            loads = [self.payload_1,self.payload_2,self.payload_3]

        if self.args.load_type == "four":
            self.payload_1 = Shape('Cuboid24')
            self.payload_2 = Shape('Cuboid25')
            self.payload_3 = Shape('Cuboid28')
            self.payload_4 = Shape('Cuboid29')

            loads = [self.payload_1,self.payload_2,self.payload_3,self.payload_4]
        
        if self.args.load_type == "six":
            self.payload_1 = Shape('Cuboid24')
            self.payload_2 = Shape('Cuboid25')
            self.payload_3 = Shape('Cuboid26')
            self.payload_4 = Shape('Cuboid27')
            self.payload_5 = Shape('Cuboid28')
            self.payload_6 = Shape('Cuboid29')

            loads = [self.payload_1,self.payload_2,self.payload_3,self.payload_4,self.payload_5,self.payload_6] 
        #####################################
        #visualization goal

        points = []

        if self.num_a == 3:
            assert(self.args.load_type=="three")
            # choice_list = random.sample(range(len(loads)),k=3)
            # # choice_list.sort() # permutation invariance
            # for i in range(len(choice_list)):
            #     points.append([loads[choice_list[i]].get_position()[0],loads[choice_list[i]].get_position()[1],1.5])
            for i in range(len(loads)):
                points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])

        # if self.num_a == 3:
        #     assert(self.args.load_type=="six")
        #     choice_list = random.sample(range(len(loads)),k=3)
        #     # choice_list.sort() # permutation invariance
        #     for i in range(len(choice_list)):
        #         points.append([loads[choice_list[i]].get_position()[0],loads[choice_list[i]].get_position()[1],1.5])

        if self.num_a == 4:
            assert(self.args.load_type=="four")
            # for i in range(len(loads)):
            #     points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])
            # i_list = random.sample(range(4),k=4)
            # for i in i_list:
            #     points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])
            for i in range(len(loads)):
                points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])

        if self.num_a == 6:
            assert(self.args.load_type=="six")
            for i in range(len(loads)):
                points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])
            
        goal_points = np.array(points)
        for i in range(self.num_a):
            self.targets[i].set_position(goal_points[i])

        return goal_points
    

    def random_position_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        saved_agents = []
        vpts = []
        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state1.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Drone_s1(i))
            if i == 0:
                self.agents[i].agent.set_3d_pose([random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max),1.7,0.0,0.0,0.0])
                vx = self.agents[i].agent.get_position()[0]
                vy = self.agents[i].agent.get_position()[1]
                vpts.append([vx,vy])
                saved_agents.append(i)
            else:
                vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                check_list = [self.check_collision_p(vpt,vpts[m]) for m in saved_agents]
                check_conditions = np.sum(check_list)

                while check_conditions:
                    vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                    check_list = [self.check_collision_p(vpt,vpts[m]) for m in saved_agents]
                    check_conditions = np.sum(check_list)

                # print("all",vpts)
                # print("current",vpt)
                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
                vpts.append(vpt)
                saved_agents.append(i)
        return model_handles,objs
    
    def direct_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"

        if self.num_a == 3:
            pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)
        
        if self.num_a == 4:
            pos = np.array([[3.5,4.5],[3.5,-4.5],[-3.5,-4.5],[-3.5,4.5]])
            # pos = np.array([[1.5,4.5],[3.5,4.5],[-1.5,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        if self.num_a == 6:
            # pos = np.array([[4.5,4.5],[4.5,-4.5],[-4.5,4.5],[-4.5,-4.5],[0,4.5],[0,-4.5]])
            pos = np.array([[4.5,4.5],[-4.5,4.5],[0,-4.5],[0,4.5],[-4.5,-4.5],[4.5,-4.5]])
            # pos = np.array([[3.5,4.5],[3.5,-4.5],[-3.5,4.5],[-3.5,-4.5],[0,4.5],[0,-4.5]])
            # pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Drone_s1(i))
            
            self.agents[i].agent.set_3d_pose([pos[i,0],pos[i,1],1.7,0.0,0.0,0.0])
                
        return model_handles,objs


    def reset_world(self):
        self.time_step = 0
        
        #self.suction_cup.release()
        if not self.close_simulation:
            for i in range(self.num_a):
                self.pr.remove_model(self.model_handles[i])
            self.pr.remove_model(self.load_handle)
        
        # self.model_handles,ii = self.direct_spread()
        self.model_handles,ii = self.random_position_spread()
        
        self.load_handle = self.load_spread()

        if self.close_simulation:
            self.targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.56, 0.56, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_a)]

        self.goals = self.generate_goal()

        # each column represents distance of all agents from the respective landmark
        world_dists = np.array([[np.linalg.norm(a.get_2d_pos() - self.goals[l,:2]) for l in range(self.goals.shape[0])]
                               	for a in self.agents])
        # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
        _, self.ci = linear_sum_assignment(world_dists)

        for j in range(self.num_a):
            self.agents[j]._reset()
    
        #for hovering when reset
        for j in range(50):
            for agent in self.agents:
                agent.hover(1.7)
            self.pr.step()
     
        obs_n = []
        for i,agent in enumerate(self.agents):
            obs_n.append(self.observation_callback(i,agent))

        self.close_simulation = False

        return np.array(obs_n)


    def step(self, action_n):

        obs_n = []
        reward_n = []
        done_n = []
        done_ter = []

        for j in range(10):
            # set action for each agent
            for i, agent in enumerate(self.agents):
                agent.set_action(action_n[i])
                # agent.set_action_pos(action_n[i], self.pr)
            self.pr.step()

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            #----------------------------
            obs_n.append(self.observation_callback(i,agent))
            rw,ter,dter = self.reward_callback(i,agent)
            reward_n.append(rw)
            done_n.append(ter)
            done_ter.append(dter)

        self.time_step+=1

        return np.array(obs_n), np.array(reward_n), np.array(done_n)

    def step_evaluate(self, action_n):

        obs_n = []
        reward_n = []
        done_n = []
        done_ter = []

        for j in range(10):
            # set action for each agent
            for i, agent in enumerate(self.agents):
                agent.set_action(action_n[i])
            self.pr.step()

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            #----------------------------
            obs_n.append(self.observation_callback(i,agent))
            rw,ter,dter = self.reward_callback(i,agent)
            reward_n.append(rw)
            done_n.append(ter)
            done_ter.append(dter)

        self.time_step+=1

        return np.array(obs_n), np.array(reward_n), np.array(done_n), np.all(done_ter)


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
        

    def reward_and_terminate(self, i, agent):
        rew = 0
        done_terminate = 0
        terminate = 0
        finish_sig = 0

        max_roll = 1.57 # Max roll after which we end the episode
        max_pitch = 1.57 # Max roll after which we end the episode

        current_orientation = agent.agent.get_orientation()

        has_flipped = True
        if current_orientation[0] > -1*max_roll and current_orientation[0] <= max_roll:
            if current_orientation[1] > -1*max_pitch and current_orientation[1] <= max_pitch:
                has_flipped = False

        #team reward
        dist = np.sqrt(np.sum(np.square(agent.get_2d_pos() - self.goals[self.ci[i],:2])))
        finish_sig = (dist < 0.5)
        rew -= dist/(self.field_size*2)
                
        if finish_sig:
            done_terminate = 1 
            # terminate = 1
            # rew = 0
            # rew += 1
            rew = 1

        #collision detection
        wall_dists = np.array([np.abs(self.field_size-agent.agent.get_position()[1]),np.abs(self.field_size+agent.agent.get_position()[1]),np.abs(self.field_size+agent.agent.get_position()[0]),np.abs(self.field_size-agent.agent.get_position()[0])]) # rangefinder: forward, back, left, right
        wall_sig = np.any(wall_dists<0.306)

        agent_collision = []
       
        for a in self.agents:
            if a == agent: continue

            if self.check_collision_a(agent,a):
                agent_collision.append(1)
            else:
                agent_collision.append(0)

        agent_sig = np.any(np.array(agent_collision))

        if wall_sig or has_flipped or agent_sig:
            rew-=1
            terminate = 1
        
        # #individual reward
        # if agent.agent.assess_collision():
        #     rew-=1
        #     terminate = 1
       
        if agent.agent.get_position()[2]<1.3:
            terminate = 1
        
        vel = agent.get_2d_vel()
        rew = rew - np.sqrt(vel[0]**2+vel[1]**2)*0.08

        return rew,terminate,done_terminate

    #local observation 
    def local_observation(self, index, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        
        entity_pos.append((self.goals[self.ci[index],:2]-agent.get_2d_pos())/(self.field_size*2))   
        proximity_read = [np.array(agent.get_promity_readings())]
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
                other_pos.append((other.get_2d_pos() - agent.get_2d_pos())/(self.field_size*2))
                other_vel.append(other.get_2d_vel())
                other_live.append([1])

        return np.round(np.concatenate([agent.get_2d_vel()]+ [agent.get_2d_pos()/self.field_size] + other_vel + other_pos + other_live + entity_pos+proximity_read),4)


    def observation(self, index, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        # for i in range(self.goals.shape[0]):  # world.entities:
        entity_pos.append((self.goals[self.ci[index],:2]-agent.get_2d_pos())/(self.field_size*2))   
        # entity_pos.append(agent.target-agent.state.p_pos)
        proximity_read = [np.array(agent.get_promity_readings())]
        
        # communication of all other agents
        other_pos = []
        for other in self.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append((other.get_2d_pos() - agent.get_2d_pos())/(self.field_size*2))

        other_vel = []
        for other in self.agents:
            if other is agent: continue
            other_vel.append(other.get_2d_vel())

        # pos_obs = agent.agent.get_position()[:2]/2.5

        return np.round(np.concatenate([agent.get_2d_vel()]+ [agent.get_2d_pos()/self.field_size] + other_vel + other_pos + entity_pos + proximity_read),4)

    


