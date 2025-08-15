#!/usr/bin/env python
# Created at 2020/3/12

import torch.nn as nn
from pyrep.networks.mlp_actor import Actor


class JointPolicy(nn.Module):
    """
    Joint Policy include:
    agent policy: (agent_state,) -> agent_action
    env policy: (agent_state, agent_action) -> agent_next_state
    """

    def __init__(self, initial_state, args):
        super(JointPolicy, self).__init__()
        self.config = args
        # self.trajectory_length = self.config.trajectory_length
        self.agent_policy = Actor(num_states=self.config.agent_num_states,
                                  num_actions=self.config.agent_num_actions,
                                  num_discrete_actions=self.config.agent_num_discrete_actions,
                                  discrete_actions_sections=self.config.agent_discrete_actions_sections,
                                  action_log_std=self.config.agent_action_log_std,
                                  use_multivariate_distribution=self.config.agent_use_multivariate_distribution,
                                  num_hiddens=self.config.agent_num_hiddens,
                                  drop_rate=self.config.agent_drop_rate,
                                  activation=self.config.agent_activation)

        # Joint policy generate trajectories sampling initial state from expert data
        self.initial_agent_state = initial_state


    def get_log_prob(self, states, actions):
        agent_action_log_prob = self.agent_policy.get_log_prob(states, actions)

        return agent_action_log_prob

