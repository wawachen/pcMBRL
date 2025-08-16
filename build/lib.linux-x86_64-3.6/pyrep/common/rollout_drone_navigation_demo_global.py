import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time

class Rollout:
    def __init__(self, args, env):
        self.args = args
        self.episode_limit = args.max_episode_len
        self.max_episodes = args.max_episodes
        self.restart_frequency = 500
        self.env = env
        self.log_path = os.getcwd()+"/log_drone3_demo"

        self.record_actions = []
        self.record_states = []
        self.record_states_next = []
        self.record_rewards = []

    def run(self):
        returns = []
        logger = SummaryWriter(logdir=self.log_path) # used for tensorboard
        score = 0

        for i in range(self.max_episodes):
            if ((i%self.restart_frequency)==0)and(i!=0):
                self.env.restart()
            s = self.env.reset_world()
            score = 0
            #for storage
            ep_states = []
            ep_actions = []
            ep_states_next = []
            ep_rewards = []

            for t in range(self.episode_limit):
                # start = time.time()
                s_next, u, r, done = self.env.step_orca() 
                
                # later = time.time()
                # print(later-start)
                # print(s_next)
                ep_states.append(s)
                ep_actions.append(u)
                ep_states_next.append(s_next)
                ep_rewards.append(r)

                score += r[0] #all reward for each agent is the same

                s = s_next

                # if np.any(done):
                #     break

            logger.add_scalar('mean_episode_rewards', score, i)
            # logger.add_scalar('network_loss', loss_sum, i)
        
            print("episode%d"%i,":",score)
            self.record_states.append(ep_states)
            self.record_actions.append(ep_actions)
            self.record_states_next.append(ep_states_next)
            self.record_rewards.append(ep_rewards)
            
        logger.close()
        return self.record_states,self.record_actions,self.record_states_next,self.record_rewards
    
    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset_world()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                s_next, r, done = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)

        return sum(returns) / self.args.evaluate_episodes
