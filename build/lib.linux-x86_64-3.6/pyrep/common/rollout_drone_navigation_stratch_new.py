from pyrep.policies.maddpg_drone_agent_scratch_new import Agent
from pyrep.common.replay_buffer import Buffer
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time

class Rollout:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.max_episodes = args.max_episodes
        self.restart_frequency = 500
        self.env = env
        self.agent = Agent(args)
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_path = os.getcwd()+"/log_drone3_stratch_new"


    def run(self):
        returns = []
        logger = SummaryWriter(logdir=self.log_path) # used for tensorboard
        score = 0

        for i in range(self.max_episodes):
            if ((i%self.restart_frequency)==0)and(i!=0):
                self.env.restart()
            s = self.env.reset_world()
            score = 0

            for t in range(self.episode_limit):
                # start = time.time()
                with torch.no_grad():                      
                    actions = self.agent.select_actions(s, self.noise, self.epsilon)
                    u = actions.copy()

                s_next, r, done = self.env.step(actions)
                # later = time.time()
                # print(later-start)
                # print(s_next)
                score += r[0] #all reward for each agent is the same
                self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
                s = s_next

                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    self.agent.learn(transitions)

                self.noise = max(0.05, self.noise - 0.0000005)
                self.epsilon = max(0.05, self.noise - 0.0000005)

                if np.any(done):
                    break

            logger.add_scalar('mean_episode_rewards', score, i)
            # logger.add_scalar('network_loss', loss_sum, i)
        
            print("episode%d"%i,":",score)
            
        logger.close()
    
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
