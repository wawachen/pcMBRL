from os import path
from os.path import dirname, join, abspath

from numpy.core.defchararray import add
from pyrep import PyRep
from pyrep.envs.drone_RL_agent import Drone_ORCA1

import numpy as np

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Drone_Env:

    def __init__(self,env_name,num_agents):
        self.time_step = 0
        # configure spaces
        self.num_a = num_agents
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles = self.import_agent_models()
        self.agents = [Drone_ORCA1(i) for i in range(num_agents)]

        r = 6.0
        dt_angle = 2*np.pi/self.num_a
        for i in range(self.num_a):
            x = r*np.cos(dt_angle*i)
            y = r*np.sin(dt_angle*i)

            self.agents[i].agent.set_3d_pose([x,y,1.7,0.0,0.0,0.0])
            self.agents[i].get_agent((x,y))
            self.agents[i].goal[0] = -x
            self.agents[i].goal[1] = -y
    
        #for hovering when reset
        for j in range(50):
            for agent in self.agents:
                agent.hover(1.7)
            self.pr.step()
    

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
        

    def step_orca(self):
        positions_n = []
        velocities_n = []
        action_n = []

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


        for i, agent in enumerate(self.agents):
            agent.set_action(action_n[i])
            # agent.set_action_pos(action_n[i], self.pr)
        self.pr.step()


        self.time_step+=1


    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()
