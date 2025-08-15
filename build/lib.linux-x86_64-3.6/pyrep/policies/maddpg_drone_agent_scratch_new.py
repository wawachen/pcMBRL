import numpy as np
import torch
import os
from pyrep.policies.maddpg_drone_scratch_new import MADDPG

class Agent:
    def __init__(self, args):
        self.args = args
        self.policy = MADDPG(args)

    def select_actions(self, o, noise_rate, epsilon):
        us = []
        for agent_id in range(self.args.n_agents):
            if np.random.uniform() < epsilon:
                u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[agent_id])
            else:
                inputs = torch.tensor(o[agent_id], dtype=torch.float32).unsqueeze(0)
                pi = self.policy.actor_networks[agent_id](inputs).squeeze(0)
                # print('{} : {}'.format(self.name, pi))
                u = pi.cpu().numpy()
                noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
                # print("noise: ", np.random.randn(*u.shape))
                u += noise
                u = np.clip(u, -self.args.high_action, self.args.high_action)
            us.append(u.copy())
        return us

    def learn(self, transitions):
        self.policy.train(transitions)
    

