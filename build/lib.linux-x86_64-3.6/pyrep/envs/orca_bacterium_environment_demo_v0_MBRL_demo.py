from os import path
from pyrep import PyRep
from pyrep.envs.drone_RL_agent import Drone_ORCA1
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

import numpy as np
from gym import spaces
import random
import math

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Drone_Env:

    def __init__(self,args,env_name,num_agents):
        ####
        self.args = args
        self.reset_callback = self.reset_world
        self.reward_callback = self.reward_and_terminate
        self.observation_callback = self.observation
        # self.done_callback = self.done

        # environment parameters
        self.discrete_action_space = False
        self.time_step = 0
        self.global_step = 0  # Global step counter for goal updates
        self.field_size = args.field_size/2

        # configure spaces
        self.num_a = num_agents
        self.env_name = env_name
        self.close_simulation = False

        self.safe_distance = 0.5
        self.x_limit_min = -self.field_size+self.safe_distance/2+0.5
        self.x_limit_max = self.field_size-self.safe_distance/2-0.5
        self.y_limit_min = -self.field_size+self.safe_distance/2+0.5
        self.y_limit_max = self.field_size-self.safe_distance/2-0.5

        self.shared_reward = True
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles,self.load_handle = self.import_agent_models()
        self.agents = [Drone_ORCA1(i,num_agents) for i in range(num_agents)]

        self.payload = Shape('Cuboid4')
        self.targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.56, 0.56, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_a)]

        self.goals = self.generate_goal()
        
        # physical action space
        u_action_space = spaces.Box(low=-1, high=1, shape=(2*num_agents,), dtype=np.float32)

        self.action_space = u_action_space
        #observation space
        obs_dim = len(self.observation_callback())
        # print(obs_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        ####
    
    def import_agent_models(self):
        robot_DIR = "/home/wawa/MARL_transport/examples/models"
        
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)

        if self.args.load_type == "three":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_three.ttm'))
        if self.args.load_type == "four":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_four.ttm'))
        if self.args.load_type == "six":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_six.ttm'))

        return model_handles,m2

    def check_collision_a(self,agent1,agent2):
        delta_pos = agent1.agent.get_drone_position()[:2] - agent2.agent.get_drone_position()[:2]
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        return True if dist <= self.safe_distance else False

    def check_collision_p(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance < 2.5:
            return 1
        else:
            return 0
    
    def load_spread(self):
        robot_DIR = "/home/wawa/MARL_transport/examples/models"  

        ################
        if self.args.load_type == "three":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_three.ttm'))
        if self.args.load_type == "four":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_four.ttm'))
        if self.args.load_type == "six":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_six.ttm'))
        ####################
        
        self.payload = Shape('Cuboid4')
        self.payload.set_orientation([0.0,0.0,0.0])
        self.payload.set_position([0.0,0.0,0.1])

        return m2

    def generate_goal(self):
        #visualization goal
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

        points = []

        ##############################################################################
        # saved_agents = []
        # vpts = []
        # for i in range(self.num_a):
        #     if i == 0:
        #         points.append([random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max),1.5])
        #         vx = points[0][0]
        #         vy = points[0][1]
        #         vpts.append([vx,vy])
        #         saved_agents.append(i)
        #     else:
        #         vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
        #         check_list = [self.check_collision_p(vpt,vpts[m]) for m in saved_agents]
        #         check_conditions = np.sum(check_list)

        #         while check_conditions:
        #             vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
        #             check_list = [self.check_collision_p(vpt,vpts[m]) for m in saved_agents]
        #             check_conditions = np.sum(check_list)

        #         points.append([vpt[0],vpt[1],1.5])
        #         vpts.append(vpt)
        #         saved_agents.append(i)
        ##############################################################################

        for i in range(len(loads)):
            points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])
            
        goal_points = np.array(points)

        for i in range(self.num_a):
            self.targets[i].set_position(goal_points[i])

        return goal_points
    
    def generate_random_goals(self):
        """Generate random goals distributed throughout the workspace"""
        points = []
        saved_goals = []
        vpts = []
        
        for i in range(self.num_a):
            if i == 0:
                goal_pos = [random.uniform(self.x_limit_min, self.x_limit_max),
                           random.uniform(self.y_limit_min, self.y_limit_max), 1.5]
                points.append(goal_pos)
                vpts.append([goal_pos[0], goal_pos[1]])
                saved_goals.append(i)
            else:
                vpt = [random.uniform(self.x_limit_min, self.x_limit_max),
                       random.uniform(self.y_limit_min, self.y_limit_max)]
                check_list = [self.check_collision_p(vpt, vpts[m]) for m in saved_goals]
                check_conditions = np.sum(check_list)

                while check_conditions:
                    vpt = [random.uniform(self.x_limit_min, self.x_limit_max),
                           random.uniform(self.y_limit_min, self.y_limit_max)]
                    check_list = [self.check_collision_p(vpt, vpts[m]) for m in saved_goals]
                    check_conditions = np.sum(check_list)

                points.append([vpt[0], vpt[1], 1.5])
                vpts.append(vpt)
                saved_goals.append(i)
        
        goal_points = np.array(points)
        
        # Update target positions
        for i in range(self.num_a):
            self.targets[i].set_position(goal_points[i])
        
        return goal_points
    
    def update_goals(self):
        """Update goals and assign them to agents"""
        self.goals = self.generate_random_goals()
        
        # Assign new goals to each agent
        for i in range(self.num_a):
            self.agents[i].goal[0] = self.goals[i][0]
            self.agents[i].goal[1] = self.goals[i][1]
        
        return self.goals


    def random_position_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/MARL_transport/examples/models"
        saved_agents = []
        vpts = []
        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Drone_ORCA1(i,self.num_a))
            if i == 0:
                self.agents[i].agent.set_3d_pose([random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max),1.7,0.0,0.0,0.0])
                vx = self.agents[i].agent.get_position()[0]
                vy = self.agents[i].agent.get_position()[1]
                vpts.append([vx,vy])
                saved_agents.append(i)
                self.agents[i].get_agent((vx,vy))
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
                self.agents[i].get_agent((vpt[0],vpt[1]))
        return model_handles,objs

    def direct_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/MARL_transport/examples/models"

        if self.num_a == 3:
            pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)
        
        if self.num_a == 4:
            pos = np.array([[3.5,4.5],[3.5,-4.5],[-3.5,-4.5],[-3.5,4.5]])
            # pos = np.array([[2.5,4.5],[3.5,4.5],[-2.5,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        if self.num_a == 6:
            pos = np.array([[3.5,4.5],[3.5,-4.5],[-3.5,4.5],[-3.5,-4.5],[0,4.5],[0,-4.5]])
            # pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Drone_ORCA1(i,self.num_a))
            
            self.agents[i].agent.set_3d_pose([pos[i,0],pos[i,1],1.7,0.0,0.0,0.0])
            self.agents[i].get_agent((pos[i,0],pos[i,1]))
                

        return model_handles,objs
        
    def reset_world(self):
        self.time_step = 0
        self.global_step = 0  # Reset global step counter
        
        #self.suction_cup.release()
        if not self.close_simulation:
            for i in range(self.num_a):
                self.pr.remove_model(self.model_handles[i])
            self.pr.remove_model(self.load_handle)

        self.model_handles,ii = self.random_position_spread()

        self.load_handle = self.load_spread()
        # self.model_handles,ii = self.direct_spread()
        if self.close_simulation:
            self.targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.56, 0.56, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_a)]

        self.goals = self.generate_goal()

        # assign goals to each agent
        # save_list = []
        # for i in range(self.num_a):
        #     dists = [np.sqrt(np.sum(np.square(a.get_2d_pos() - self.goals[i,:2]))) if i not in save_list else 1000 for i,a in enumerate(self.agents)]
        #     min_d = min(dists)
        #     save_list.append(dists.index(min_d))

        # for i in range(self.num_a):
        #     self.agents[save_list[i]].goal[0] = self.goals[i][0]
        #     self.agents[save_list[i]].goal[1] = self.goals[i][1]

        for i in range(self.num_a):
            self.agents[i].goal[0] = self.goals[i][0]
            self.agents[i].goal[1] = self.goals[i][1]
        
        for j in range(self.num_a):
            self.agents[j]._reset()
    
        #for hovering when reset
        for j in range(50):
            for agent in self.agents:
                agent.hover(1.7)
            self.pr.step()
        
        obs_n = self.observation_callback()

        self.close_simulation = False

        return np.array(obs_n)

    #This function is used for the demonstration orca environment
    def step_orca(self):
        # Check if it's time to update goals (every 10 steps)
        if self.global_step % 10 == 0:
            self.update_goals()
            print(f"Goals updated at step {self.global_step}")
        
        positions_n = []
        velocities_n = []
        action_n = []

        #for recording
        obs_n = []
        done_n = []

        for i, agent in enumerate(self.agents):
            for other in self.agents:
                if other is agent:
                    continue
                positions_n.append(np.array([other.get_2d_pos()[0],other.get_2d_pos()[1]]))
                velocities_n.append(np.array([other.get_2d_vel()[0],other.get_2d_vel()[1]]))
            
            agent.update_agent()
            agent.computeNeighbors(positions_n,velocities_n)
            action_n.append(agent.computeNewVelocity1())

            positions_n = []
            velocities_n = []

        #######
        for j in range(10):
            # set action for each agent
            for i, agent in enumerate(self.agents):
                agent.set_action(action_n[i])
                # agent.set_action_pos(action_n[i], self.pr)
            self.pr.step()
        #######

        obs_n = self.observation_callback()
        # record observation for each agent
        for i, agent in enumerate(self.agents):
            #----------------------------
            ter = self.reward_callback(agent)
            done_n.append(ter)

        self.time_step+=1
        self.global_step+=1  # Increment global step counter

        return np.array(obs_n), np.concatenate(action_n), np.array(done_n)


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

    #This function is used for the demonstration orca environment
    def reward_and_terminate(self, agent):
        terminate = 0 

        #collision detection
        wall_dists = np.array([np.abs(self.field_size-agent.agent.get_position()[1]),np.abs(self.field_size+agent.agent.get_position()[1]),np.abs(self.field_size+agent.agent.get_position()[0]),np.abs(self.field_size-agent.agent.get_position()[0])]) # rangefinder: forward, back, left, right
        wall_sig = np.any(wall_dists<0.206)

        agent_collision = []
        for a in self.agents:
            if a == agent: continue
            if self.check_collision_a(agent,a):
                agent_collision.append(1)
            else:
                agent_collision.append(0)
        agent_sig = np.any(np.array(agent_collision))

        if agent_sig or wall_sig:   
            terminate = 1
       
        if agent.agent.get_position()[2]<1.3:
            terminate = 1

        return terminate

    def observation(self):
        # communication of all other agents
        all_pos = []
        for agent in self.agents:
            all_pos.append(agent.get_2d_pos()/self.field_size)

        all_vel = []
        for agent in self.agents:
            all_vel.append(agent.get_2d_vel())

        all_ori = []
        for agent in self.agents:
            all_ori.append(agent.get_orientation())

        return np.concatenate(all_pos+all_vel+all_ori)


