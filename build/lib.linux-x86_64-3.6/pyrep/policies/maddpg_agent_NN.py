import numpy as np
import torch
import os
from pyrep.policies.maddpg_NN import MADDPG

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u1 = np.random.uniform(-1,1)
            u2 = np.random.uniform(0,1)
            u = np.array([u1,u2])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * np.random.randn(*u.shape)  # gaussian noise
            # print("noise: ", np.random.randn(*u.shape))
            u += noise
            u[0] = np.clip(u[0], -1, 1)
            u[1] = np.clip(u[1], 0, 1)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

