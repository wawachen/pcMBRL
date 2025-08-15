from pyrep.robots.mobiles.new_quadricopter import NewQuadricopter
import numpy as np
from pyrep.envs.array_specs import ArraySpec

class Drone:

    def __init__(self, id):
        
        #action: roll, pitch, thrust of UAV      
        self._action_spec = ArraySpec(shape=(3,), dtype=np.float32)
        #observations: postion, oriention,depth<1>, proximity sensor<4>, linear velocity<2>
        self._observation_spec = ArraySpec(
            shape=(6,), #13
            dtype=np.float32,
        )
        #minimum=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], maximum=[2.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
        self.agent = NewQuadricopter(0,4)
        self.target_z = 2.0
        self.max_duration = 125
        #self.suction_cup = UarmVacuumGripper()
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_state(self):
        #vector_state = np.r_[self.agent.get_drone_position()[:], self.agent.get_orientation()[:],self.agent.get_concentration(), [self.agent.get_front_proximity(),self.agent.get_back_proximity(),self.agent.get_left_proximity(), self.agent.get_right_proximity()], self.agent.get_velocities()[0][:2]]
        vector_state = np.r_[self.agent.get_drone_position()[:], self.agent.get_orientation()[:]]
        vector_state1 = vector_state.astype(np.float32)
        return vector_state1

    def _reset(self,px,py):
        #self.suction_cup.release()
        #---takeoff
        self.agent.drone_reset()
        self.agent.set_3d_pose([px,py,0.0,0.0,0.0,0.0])


    def hover(self,pos):
        vels = self.agent.position_controller([pos[0],pos[1],1.7])
        self.agent.set_propller_velocity(vels)

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)


    def _step(self, action,timestamp):
        vels = self.agent.rpythrust_controller(action)
        self.agent.set_propller_velocity(vels[:])

        self.agent.control_propeller_thrust(1)
        self.agent.control_propeller_thrust(2)
        self.agent.control_propeller_thrust(3)
        self.agent.control_propeller_thrust(4)
        
        #self.suction_cup.grasp(target)
        #print('wawa')
        self._state = self._get_state()
        #for landing task:
        # linear_acceleration_z = self.agent.get_IMUdata()[0][2]

        #detect = self.suction_cup.check_connection(self.target)
        #depth = self._state[6]
        
        # scanning_data = self._state[3:7] #up down left right
        # vels = self._state[7:]
       #take off task---------------------------------------------------------------------------
        # pos = self._state[:3]
        # done = False
        # reward = -min(abs(self.target_z-pos[2]), 4.0)

        # if pos[2] >= self.target_z:  # agent has crossed the target height
        #     reward += 10.0  # bonus reward
        #     done = True
        # elif timestamp > self.max_duration:  # agent has run out of time
        #     reward -= 10.0  # extra penalty
        #     done = True

        #hovering task--------------------------------------------------------------------------
        pos = self._state[:3]
        done = False
        reward = -abs(self.target_z-pos[2])

        def is_equal(x, y, delta=0.0):
            return abs(x-y) <= delta

        if is_equal(self.target_z, pos[2], delta=0.1):
            reward += 10.0  # bonus rewardself.get_distance(agent.state.p_pos, world.landmarks[0].state.p_pos)
            # done = True

        if timestamp > self.max_duration:
            # reward -= 10.0  # extra penalty
            done = True

        #landing task---------------------------------------------------------------------------
        #change target z
        # pos = self._state[:3]
        # done = False
        # reward = -abs(self.target_z - pos[2])
        # reward += -abs(linear_acceleration_z)
        # if pos[2] == self.target_z:
        #     reward += 10.0  # bonus reward
        #     done = True
        
        # if timestamp > self.max_duration:
        #     reward -= 10.0  # extra penalty
        #     done = True

        return np.r_[self._state, reward,done]
