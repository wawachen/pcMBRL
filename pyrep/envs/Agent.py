from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import torch
from scipy.io import savemat

TORCH_DEVICE = torch.device('cuda')


class Agent:
    """An general class for RL agents.
    """

    def __init__(self, env):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        self.env = env

    def sample(self, horizon, policy): 
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """

        times, rewards = [], []
        
        s = self.env.reset_world()
        O, A, reward_sum, done = [], [], 0, False
        O_next = []
        top_act_seq = []
        success_list = []

        policy.reset()
        
        for t in range(horizon):
            start = time.time()

            action,act_l,store_top_s,store_bad_s, cost = policy.act(s, t, self.env.goals[:,:2]) #[6,5,2] store top s
            
            A.append(action)
            top_act_seq.append(act_l)
            times.append(time.time() - start)

            s_next, reward, done, success = self.env.step(A[t])

            # print("action",A[t])

            O.append(s)
            O_next.append(s_next)
            reward_sum += reward[0]
            rewards.append(reward[0])
            success_list.append(success)

            s = s_next

            if done or (np.sum(success_list) == 10):
                break
        
        # print("Average action selection time: ", np.mean(times))
        # print("Rollout length: ", len(A))
        # print("Rollout reward: ", reward_sum)

        return {
            "obs": np.array(O),
            'obs_next': np.array(O_next),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "reward_average": reward_sum/len(A),
            "rewards": np.array(rewards)
        }