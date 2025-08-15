#!/usr/bin/env python
# Created at 2020/3/10
import numpy as np
from torch.utils.data import Dataset

from pyrep.common.torch_util import FLOAT


class ExpertDataSet(Dataset):
    def __init__(self, data_set_path, n_agent):
        demoData = np.load(data_set_path)
    
        """episode_batch:  self.demo_episode x timestep x n x keydim   
        """
        action_episodes = demoData['acs']
        obs_episodes = demoData['obs']
        obs_next_episodes = demoData['obs_next']

        assert(action_episodes.shape[0]==500)
        assert(action_episodes.shape[1]==25)

        if n_agent == 3:
            demo_timesteps = 12
        if n_agent == 4:
            demo_timesteps = 25
        if n_agent == 6:
            demo_timesteps = 35

        self.state = []
        self.action = []
        self.next_state = []
        for i in range(action_episodes.shape[0]):
            for j in range(demo_timesteps):
                state_temp = []
                action_temp = []
                next_state_temp = []
                for m in range(n_agent):
                    state_temp.append(obs_episodes[i,j,m,:])
                    action_temp.append(action_episodes[i,j,m,:])
                    next_state_temp.append(obs_next_episodes[i,j,m,:])
                # print(state_temp)
                self.state.append(np.concatenate(state_temp,axis=0))
                self.action.append(np.concatenate(action_temp,axis=0))
                self.next_state.append(np.concatenate(next_state_temp,axis=0))
        
        self.state = FLOAT(np.array(self.state))
        self.action = FLOAT(np.array(self.action))
        self.next_state = FLOAT(np.array(self.next_state))

        self.length = self.state.shape[0]
        
        assert self.state.shape[1]==n_agent*obs_episodes.shape[3] 
    

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.state[idx], self.action[idx]
