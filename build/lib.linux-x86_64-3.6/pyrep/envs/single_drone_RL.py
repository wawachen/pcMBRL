from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.envs.drone_agent1 import Drone

from pyrep.policies.buffer import ReplayBuffer
from pyrep.policies.Sddpg import SDDPG

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
# from pyrep.robots.end_effectors.uarm_Vacuum_Gripper import UarmVacuumGripper
import matplotlib
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc
import cv2
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

from collections import OrderedDict
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from pyrep.policies.utilities import transpose_list, transpose_to_tensor, hard_update, soft_update
from collections import deque
import progressbar as pb
import math

class Single_Drone_Env:
  def __init__(self,env_name):
      fps = 2
      self.pr = PyRep()
      self.pr.launch(env_name, headless=False)
      self.pr.start()

      self.model_handle = self.import_agent_model()
      self.agent = Drone(0) 

      self.num_dd = self.agent.observation_spec().shape[0]
      
      #self.suction_cup = UarmVacuumGripper()
      # self.target = Shape('Cuboid')

      # self.random_position_spread()

      self.cam = VisionSensor('Video_recorder')
      fourcc = VideoWriter_fourcc(*'MP42')
      self.video = VideoWriter('./my_vid_test_single.avi', fourcc, float(fps), (self.cam.get_resolution()[0], self.cam.get_resolution()[1]))

  def import_agent_model(self):
      robot_DIR = "/home/wawa/RL_cotransportation/examples/models"
      
      #pr.remove_model(m1)
      [m,m1]= self.pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))
      return m1

  def action_spec(self):
      return self.agent.action_spec()

  def observation_spec(self):
      return self.agent.observation_spec()

  def _get_state(self):
      """ To get each agent's state 
      using vector_state[i,:] i=0....n """

      total_vector = []
      
      total_vector.append(self.agent._get_state())
      vector_state = np.r_[total_vector]
      return vector_state

  def _reset(self):
      #self.target = Shape('Cuboid')
      #self.suction_cup.release()
      
      self.pr.remove_model(self.model_handle)

      self.model_handle = self.import_agent_model()
      self.agent = Drone(0)

      #self.target.set_position([1.2,0.0,0.2])
      #self.target.set_orientation([0.0, 0.0, 0.0])
      # self.random_position_spread()
      #self.spread()
      self.agent.agent.set_3d_pose([0,0,0.156,0.0,0.0,0.0])

      img = (self.cam.capture_rgb() * 255).astype(np.uint8)
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      self.video.write(img_rgb)

      #for hovering when reset
      # for _ in range(100):
      #     for i in range(self.num_a):
      #         pos = self.agents[i].agent.get_drone_position()
      #         self.agents[i].hover(pos)
      #     self.pr.step()

      # self._state = self._get_state()
      # self._episode_ended = False
      # self.step_counter = 0
      # return np.r_[states,rewards,dones]

  def _step(self, action,timestamp):
      """The dimension of action is <num,2>"""
      # if self._episode_ended:
      #     self.step_counter = 0
      #     return self.reset()
      for _ in range(2):
        step_ob = agent._step(action,timestamp)

        state = step_ob[:self.num_dd]
        reward = step_ob[self.num_dd]
        done = step_ob[self.num_dd+1]

        img = (self.cam.capture_rgb() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.video.write(img_rgb)
        self.pr.step()  #Step the physics simulation

      return [state,reward,done]

  def shutdown(self):
      self.pr.stop()
      self.pr.shutdown()


env_name = join(dirname(abspath(__file__)), 'cooperative_transportation_uav.ttt')
env = Single_Drone_Env(env_name)

action_size = env.action_spec().shape[0]
state_size = env.observation_spec().shape[0]

#parameter settings
## Initialise parameters ##
p = OrderedDict()

## Environment_parameters ##
p.update(action_size=action_size, action_type = 'continuous', state_size=state_size)

## Episode_parameters ##
p.update(number_of_episodes=10000, episode_length=100, episodes_before_training=300,
                                 learn_steps_per_env_step=3, catchup_tau=.01, catchup_threshold=1.15)
## Replay_Buffer_parameters ##
p.update(buffer_size=100000, n_steps =5)

## Agent_parameters ##
p.update(discount_rate=0.99, tau=0.0001, lr_actor=0.00025, lr_critic=0.0005)

## Model_parameters ##
p.update(batchsize=256, hidden_in_size=300, hidden_out_size=200, l2_decay=0.0001)

## Categorical_parameters ##
p.update(num_atoms=51, vmin=-0.1, vmax=1)

## Noise_parameters ##
p.update(noise_type='BetaNoise', noise_reduction=0.998 , noise_scale_end=0.001,
                               OU_mu=0, OU_theta=0.2, OU_sigma=0.2)

random_seed = np.random.randint(1000)

# set the random seed - this allows for reproducibility
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# how many episodes to save network weights
save_interval = 500
t = 0


# amplitude of noise
# this slowly decreases to 0
noise_reduction = p['noise_reduction'] # each episode we decay the noise by this
noise_scale = 1 # we start the noise at 1
noise_scale_end = p['noise_scale_end'] # the noise will never drop below this

# some performance metrics to keep track of for graphing afterwards
agent_scores = [] 
agent_scores_last_100 = [deque(maxlen = 100)]
agent_scores_avg, previous_agent_scores_avg = 0, 0

log_path = os.getcwd()+"/log1" # we save tensorboard logs here
model_dir= os.getcwd()+"/model_dir1" # we save the model files here, to be reloaded for watching the agents

os.makedirs(model_dir, exist_ok=True) # make the directory if it doesn't exist

# keep 50000 timesteps worth of replay, with n-step-5 bootstraping, and 0.99 discount rate
buffer = ReplayBuffer(size = p['buffer_size'], n_steps = p['n_steps'], discount_rate = p['discount_rate'])

# initialize actor and critic networks and ddpg agents, passing all parameters to it
sddpg = SDDPG(p)

logger = SummaryWriter(logdir=log_path) # used for tensorboard

# training loop
# show progressbar of several metrics of interest
# all the metrics progressbar will keep track of
widget = ['episode: ', pb.Counter(),'/',str(p['number_of_episodes']),' ',
            pb.DynamicMessage('a0_avg_score'), ' ',
            pb.DynamicMessage('a0_noise_scale'), ' ',
            pb.DynamicMessage('buffer_size'), ' ',
            pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ] 


timer = pb.ProgressBar(widgets=widget, maxval=p['number_of_episodes']).start() # progressbar

for episode in range(0,p['number_of_episodes']):
    training_flag = episode >= p['episodes_before_training'] # the training flag denotes whether to begin training
    if training_flag: # if training
        if agent_scores_avg > previous_agent_scores_avg: # if the score is improving
            noise_scale = max(noise_scale*noise_reduction,noise_scale_end) # then reduce noise

    timer.update(episode, a0_avg_score=agent_scores_avg[0], a1_noise_scale=noise_scale[0], buffer_size=len(buffer)) # progressbar

    buffer.reset() # reseting the buffer is neccessary to ensure the n-step bootstraps reset after each episode
    reward_this_episode = 0 # keeping track of episode reward
    # if training_flag: # useful for viewing the agent in full screen, otherwise leave as is
    #     env_info = env.reset(train_mode=True)[brain_name]
    # else:
    #     env_info = env.reset(train_mode=True)[brain_name]
    env._reset()   
    states = env._get_state() # we get states directly from the environment
    obs = states # and reshape them into a list

    # save model or not this episode
    save_info = ((episode) % save_interval < 1 or episode==p['number_of_episodes']-1)
    
    
    for episode_t in range(p['episode_length']):
        
        t += 1 # increment timestep counter
        # explore only for a certain number of episodes
        # action input needs to be transposed
        # actions = maddpg.act(transpose_to_tensor([obs]), noise_scale=noise_scale) # the actors actions
        # actions_array = torch.stack(actions).detach().numpy().squeeze() # converted into a np.array
        # print(obs)
        if training_flag:
            #action is the roll pitch and thrust
            actions = sddpg.act(transpose_to_tensor([obs]), noise_scale=noise_scale) # the actors actions
            actions_array = torch.stack(actions).detach().numpy().squeeze() # converted into a np.array
            random_roll = math.radians(actions_array[0]*25) 
            random_pitch = math.radians(actions_array[1]*25) 
            thrust = (actions_array[2]+2)*5
           
            actions_array_cmd = np.r_[(random_roll,random_pitch,thrust)] # behave randomly before training

            # print(actions_array)
        else:
            random_roll = math.radians(np.random.uniform(-25, 25))
            #convert into radian
            random_pitch = math.radians(np.random.uniform(-25, 25))
            #generate velocity cmd
            thrust = np.random.uniform(5, 15)

            actions_array_cmd = np.r_[(random_roll,random_pitch,thrust)] # behave randomly before training

            
        #print(actions_array)
        env_info = env._step(actions_array_cmd)   # input the actions into the env

        next_states = env_info[0] # get the next states
        next_obs = next_states # and reshape them into a list 

        rewards = env_info[1] # get the rewards
        dones = env_info[2] # and whether the env is done

        # add data to buffer
        transition = ([obs, actions_array, rewards, next_obs, dones])
        buffer.push(transition)

        obs = next_obs # after each timestep update the obs to the new obs before restarting the loop
        
        # for calculating rewards for this particular episode - addition of all time steps
        reward_this_episode += rewards
        previous_agent_scores_avg = agent_scores_avg
        
        if dones:                                  # exit loop if episode finished
            break
            
    # update the episode scores being kept track of - episode score, last 100 scores, and rolling average scores
    
    previous_agent_scores_avg = agent_scores_avg
    agent_scores.append(reward_this_episode)
    agent_scores_last_100.append(reward_this_episode)
    agent_scores_avg = np.mean(agent_scores_last_100)

    # update agents networks
    if (len(buffer) > p['batchsize']) & training_flag:
        for _ in range(p['learn_steps_per_env_step']): # learn multiple times at every step
            
            if agent_scores_avg < (p['catchup_threshold']*min(agent_scores_avg)+0.01):# if agent too far ahead then wait
                samples = buffer.sample(p['batchsize']) # sample the buffer
                sddpg.update(samples, logger=logger) # update the agent
                sddpg.update_targets() # soft update the target network towards the actual networks
                #if t % C == 0:
                #    maddpg.hard_update_targets(agent_num) # hard update the target network towards the actual networks
                # this can be used instead of soft updates
            
            # else:
            #     # update the target networks of the worse agent towards the better one
            #     soft_update(sddpg.Sddpg_agent[1-agent_num].actor,maddpg.maddpg_agent[agent_num].actor,p['catchup_tau'])
            #     soft_update(sddpg.Sddpg_agent[1-agent_num].critic,maddpg.maddpg_agent[agent_num].critic,p['catchup_tau'])
    
    # add average score to tensorboard
    if (episode % 100 == 0) or (episode == p['number_of_episodes']-1):
        for a_i, avg_rew in enumerate(agent_scores_avg):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)
            
    '''
    #saving model
    save_dict_list =[]
    if save_info:
        for i in range(num_agents):
            save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                         'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                         'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                         'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
            save_dict_list.append(save_dict)
            torch.save(save_dict_list, os.path.join(model_dir, 'episode-{}.pt'.format(episode)))
    '''
env.shutdown()
logger.close()
timer.finish()