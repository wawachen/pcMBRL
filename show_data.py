import numpy as np
from scipy.io import savemat

n_agents = 6

# load trajectory
demoData = np.load('orca_demonstration_ep100_6agents_env10.npz')

"""episode_batch:  self.demo_episode x timestep x n x keydim   
"""
action_episodes = demoData['acs']
obs_episodes = demoData['obs']
obs_next_episodes = demoData['obs_next']

assert(action_episodes.shape[0]==100)

if n_agents == 3:
    demo_timesteps = 25
if n_agents == 4:
    demo_timesteps = 25
if n_agents == 6:
    demo_timesteps = 35

expert_state = []
expert_action = []
expert_next_state = []

for i in range(action_episodes.shape[0]):
    for j in range(demo_timesteps):
        state_temp = []
        action_temp = []
        next_state_temp = []
        for m in range(n_agents):
            state_temp.append(obs_episodes[i,j,m,:])
            action_temp.append(action_episodes[i,j,m,:])
            next_state_temp.append(obs_next_episodes[i,j,m,:])
        # print(state_temp)
        expert_state.append(np.concatenate(state_temp,axis=0))
        expert_action.append(np.concatenate(action_temp,axis=0))
        expert_next_state.append(np.concatenate(next_state_temp,axis=0))

expert_state = np.array(expert_state)
expert_action = np.array(expert_action)
expert_next_state = np.array(expert_next_state)
expert_traj = np.concatenate([expert_state,expert_action],axis = 1)

savemat('./storeState_action6.mat', mdict={'expert_sc': expert_traj})
