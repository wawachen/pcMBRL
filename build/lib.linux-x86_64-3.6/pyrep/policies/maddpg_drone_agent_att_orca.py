import numpy as np
import torch
import os
from pyrep.policies.maddpg_drone_att_orca import MADDPG

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
           u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            # inputs = torch.from_numpy(o).unsqueeze(0)
            # inputs = inputs.type(torch.float32)
            # print(inputs.shape)
            # print(torch.from_numpy(o).unsqueeze(0).shape)
            # print(torch.isfinite(inputs).all())
            # print(torch.isnan(inputs).any())
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            # print("noise: ", np.random.randn(*u.shape))
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)
    
    def learn_orca(self, transitions, other_agents):
        self.policy.train_ORCA(transitions, other_agents)

