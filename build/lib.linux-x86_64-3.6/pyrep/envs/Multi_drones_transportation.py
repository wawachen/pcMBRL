from os import path
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.envs.drone_agent import Drone

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np

from cv2 import VideoWriter, VideoWriter_fourcc
import cv2
from pyrep.objects.vision_sensor import VisionSensor
import random
import math


class Drone_Env:

    def __init__(self,env_name,num_agents):

        #Settings for video recorder 
        fps = 2

        self.safe_distance = 0.71
        self.x_limit_min = -10+self.safe_distance/2
        self.x_limit_max = 30-self.safe_distance/2
        self.y_limit_min = -10+self.safe_distance/2
        self.y_limit_max = 10-self.safe_distance/2
        self.num_a = num_agents
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles = self.import_agent_models()
        self.agents = [Drone(i) for i in range(num_agents)]

        self.num_dd = self.agents[0].observation_spec().shape[0]
        
        #self.suction_cup = UarmVacuumGripper()
        # self.target = Shape('Cuboid')

        # self.random_position_spread()

        self.cam = VisionSensor('Video_recorder')
        fourcc = VideoWriter_fourcc(*'MP42')
        self.video = VideoWriter('./my_vid_test.avi', fourcc, float(fps), (self.cam.get_resolution()[0], self.cam.get_resolution()[1]))

        # self._episode_ended = False
        # self.step_counter = 0

    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        #pr.remove_model(m1)
        [m,m1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        [m_1,m1_1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        [m_2,m1_2]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        [m_3,m1_3]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        [m_4,m1_4]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
        [m_5,m1_5]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))

        model_handles = [m1,m1_1,m1_2,m1_3,m1_4,m1_5]
        return model_handles

    def check_collision(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance <= self.safe_distance:
            return 1
        else:
            return 0

    def spread(self):
        self.agents[0].agent.set_3d_pose([15,-6,1.7,0.0,0.0,0.0])
        self.agents[1].agent.set_3d_pose([15,0,1.7,0.0,0.0,0.0])
        self.agents[2].agent.set_3d_pose([15,6,1.7,0.0,0.0,0.0])
        self.agents[3].agent.set_3d_pose([20,-6,1.7,0.0,0.0,0.0])
        self.agents[4].agent.set_3d_pose([20,0,1.7,0.0,0.0,0.0])
        self.agents[5].agent.set_3d_pose([20,6,1.7,0.0,0.0,0.0])

    def random_position_spread(self):
        saved_agents = []
        vpts = []
        for i in range(self.num_a):
            if i == 0:
                self.agents[i].agent.set_3d_pose([random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max),1.7,0.0,0.0,0.0])
                vx = self.agents[i].agent.get_drone_position()[0]
                vy = self.agents[i].agent.get_drone_position()[1]
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

                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
                vpts.append(vpt)
                saved_agents.append(i)

    def action_spec(self):
        return self.agents[0].action_spec()

    def observation_spec(self):
        return self.agents[0].observation_spec()

    def _get_state(self):
        """ To get each agent's state 
        using vector_state[i,:] i=0....n """

        total_vector = []
        for i in range(self.num_a):
            total_vector.append(self.agents[i]._get_state())
        vector_state = np.r_[total_vector]
        return vector_state

    def _reset(self):
        self.target = Shape('Cuboid')
        #self.suction_cup.release()
        for i in range(self.num_a):
            self.pr.remove_model(self.model_handles[i])

        self.model_handles = self.import_agent_models()
        self.agents = [Drone(i) for i in range(self.num_a)]

        self.target.set_position([1.2,0.0,0.2])
        self.target.set_orientation([0.0, 0.0, 0.0])
        # self.random_position_spread()
        self.spread()

        img = (self.cam.capture_rgb() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.video.write(img_rgb)

        #for hovering when reset
        for _ in range(100):
            for i in range(self.num_a):
                pos = self.agents[i].agent.get_drone_position()
                self.agents[i].hover(pos)
            self.pr.step()

        # self._state = self._get_state()
        # self._episode_ended = False
        # self.step_counter = 0
        # return np.r_[states,rewards,dones]

    def _step(self, action):
        """The dimension of action is <num,2>"""
        # if self._episode_ended:
        #     self.step_counter = 0
        #     return self.reset()
        for _ in range(20):
            states = np.zeros([self.num_a,self.num_dd])
            rewards = np.zeros(self.num_a)
            dones = np.zeros(self.num_a)

            # self.step_counter+= 1

            for i,agent in enumerate(self.agents):
                step_ob = agent._step(action[i,:])
                states[i,:] = step_ob[:self.num_dd]
                rewards[i] = step_ob[self.num_dd]
                dones[i] = step_ob[self.num_dd+1]

            img = (self.cam.capture_rgb() * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.video.write(img_rgb)
            self.pr.step()  #Step the physics simulation
        # if self.step_counter>1200:
        #     self._episode_ended = True
        
        return [states,rewards,dones]

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()