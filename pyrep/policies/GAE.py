#!/usr/bin/env python
# Created at 2020/3/10
import torch

from pyrep.common.torch_util import FLOAT, device


def estimate_advantages(rewards, masks, values, next_values, gamma, tau, done):
    """
    General advantage estimate
    :param rewards: [trajectory length * 1, 1]
    :param masks: [trajectory length * 1, 1]
    :param values: [trajectory length * 1, 1]
    :param gamma:
    :param tau:
    :param trajectory_length: the length of trajectory
    :return:
    """
    # trans_shape_func = lambda x: x.reshape(trajectory_length, -1, 1)
    # rewards = trans_shape_func(rewards)  # [trajectory length, parallel size, 1]
    # masks = trans_shape_func(masks)  # [trajectory length, parallel size, 1]
    # values = trans_shape_func(values)  # [trajectory length, parallel size, 1]

    # deltas = FLOAT(rewards.size()).to(device)
    # advantages = FLOAT(rewards.size()).to(device)
    advantages = []
    gae = 0

    # calculate advantages in parallel
    # prev_value = torch.zeros((rewards.size(1), 1), device=device)
    # prev_advantage = torch.zeros((rewards.size(1), 1), device=device)

    deltas = rewards + gamma * next_values * masks - values

    for delta, d in zip(reversed(deltas.flatten().cpu().numpy()),reversed(done.flatten().cpu().numpy())):
        gae = delta + gamma * tau * gae * d
        advantages.insert(0,gae)
    advantages = torch.tensor(advantages, dtype=torch.float).view(-1, 1)
    returns = values + advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    # reverse shape for ppo
    return advantages.reshape(-1, 1), returns.reshape(-1, 1)  # [trajectory length * parallel size, 1]
