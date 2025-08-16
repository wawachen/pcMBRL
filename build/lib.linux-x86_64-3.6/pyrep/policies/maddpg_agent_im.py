import numpy as np
import torch
import os
from pyrep.policies.maddpg_im import MADDPG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, args):
        self.args = args
        self.policy = MADDPG(args, 0)

    def select_action(self, o, goal, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u1 = np.random.uniform(-1,1)
            u2 = np.random.uniform(0,1)
            u = np.array([u1,u2])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            goal = torch.tensor(goal,dtype=torch.float32).unsqueeze(0).to(device)
            # print(inputs.shape,goal.shape)
            pi = self.policy.actor_network(inputs,goal).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            # print(u)
            noise = noise_rate * np.random.randn(*u.shape)  # gaussian noise
            # print("noise: ", np.random.randn(*u.shape))
            u += noise
            u[0] = np.clip(u[0], -1, 1)
            u[1] = np.clip(u[1], 0, 1)
        return u.copy()

    def learn(self, transitions):
        self.policy.train(transitions)

