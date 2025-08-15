from pyrep.robots.mobiles.new_quadricopter import NewQuadricopter
import numpy as np
from pyrep.envs.array_specs import ArraySpec

class Drone:

    def __init__(self, id):
        self.safe_distance = 0.71
        #action: linear velocities (vx,vy) of UAV      
        self._action_spec = ArraySpec(shape=(2,), dtype=np.float32)
        #observations: postion, oriention,depth<1>, proximity sensor<4>, linear velocity<2>
        self._observation_spec = ArraySpec(
            shape=(9,), #13
            dtype=np.float32,
        )
        #minimum=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], maximum=[2.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
        self.agent = NewQuadricopter(id,4)
        #self.suction_cup = UarmVacuumGripper()
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_state(self):
        #vector_state = np.r_[self.agent.get_drone_position()[:], self.agent.get_orientation()[:],self.agent.get_concentration(), [self.agent.get_front_proximity(),self.agent.get_back_proximity(),self.agent.get_left_proximity(), self.agent.get_right_proximity()], self.agent.get_velocities()[0][:2]]
        vector_state = np.r_[self.agent.get_drone_position()[:], [self.agent.get_front_proximity(),self.agent.get_back_proximity(),self.agent.get_left_proximity(), self.agent.get_right_proximity()], self.agent.get_velocities()[0][:2]]
        vector_state1 = vector_state.astype(np.float32)
        return vector_state1

    def _reset(self,px,py):
        #self.suction_cup.release()
        self.agent.drone_reset()
        self.agent.set_3d_pose([px,py,1.7,0.0,0.0,0.0])

    def hover(self,pos):
        vels = self.agent.position_controller([pos[0],pos[1],1.7])
        self.agent.set_propller_velocity(vels)

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)


    def _step(self, action):
        vels = self.agent.velocity_controller(action[0],action[1])
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)
        
        #self.suction_cup.grasp(target)
        
        self._state = self._get_state()
        #detect = self.suction_cup.check_connection(self.target)
        #depth = self._state[6]
        
        scanning_data = self._state[3:7] #up down left right
        vels = self._state[7:]
        pos = self._state[:3]
        #ori = self._state[3:6]
    
    #     #punish for crashing
    #     if pos[2]<0.3:
    #         ground_reward = -25
    #     else:
    #         ground_reward = 0.0

    #    # orientation_reward = 0.2 if np.power(ori[:2],2).sum()<0.06 else -5 
        
    #     if depth < 1.5:
    #         reward_depth = 30*(1.7-depth)/1.7
    #     else:
    #         reward_depth = 0.0
        
    #     if scanning_data[0]<self.safe_distance or scanning_data[1]<self.safe_distance or scanning_data[2]<self.safe_distance or scanning_data[3]<self.safe_distance:
    #         obstacle_reward = -5
    #     else:
    #         obstacle_reward = 0.0

    #     reward_sum = reward_depth + obstacle_reward+ground_reward

    #     if pos[2]<0.3 or pos[0]>31.0 or pos[0]<-11 or pos[1]<-11 or pos[1]>11:
    #         done = 1
    #     else:
    #         done = 0after a transition each UAV would get a constant penalty rs(st, at) = −3. The
         #punish for crashing
        if pos[2]<0.3:
            ground_reward = -25
        else:
             ground_reward = 0.0

        if scanning_data[0]<self.safe_distance or scanning_data[1]<self.safe_distance or scanning_data[2]<self.safe_distance or scanning_data[3]<self.safe_distance:
            obstacle_reward = -5
        else:
            obstacle_reward = 0.0

        reward_sum = -(pos[0]-1.2)**2 - (pos[1])**2 - 0.001*np.sqrt(action[0]**2+action[1]**2) + obstacle_reward+ground_reward
        if pos[2]<0.3 or pos[0]>31.0 or pos[0]<-11 or pos[1]<-11 or pos[1]>11:
             done = 1
        else:
             done = 0

        return np.r_[self._state, reward_sum,done]
