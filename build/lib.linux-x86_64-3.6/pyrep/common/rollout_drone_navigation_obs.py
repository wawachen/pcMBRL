from pyrep.policies.maddpg_agent_im import Agent
from pyrep.common.replay_buffer_im import Buffer
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class Rollout:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.max_episodes = args.max_episodes
        self.restart_frequency = 500
        self.env = env
        self.agent = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_path = os.getcwd()+"/log_drone1_obs"

    def _init_agents(self):
        agent = Agent(self.args)
        return agent

    def run(self):
        returns = []
        logger = SummaryWriter(logdir=self.log_path) # used for tensorboard
        score = 0

        for i in range(self.max_episodes):
            if ((i%self.restart_frequency)==0)and(i!=0):
                self.env.restart()
            s,g = self.env.reset_world()
            score = 0

            for t in range(self.episode_limit):
                u = []
                actions = []
                latents = []
                with torch.no_grad():
                    for agent_id in range(self.args.n_agents):
                        if agent_id == 0:
                            action = self.agent.select_action(s[0],g[0],self.noise, self.epsilon)
                            # print(action)
                            u.append(action)
                            actions.append(action)
                        elif agent_id == 1:
                            r1 = 1.0
                            action = np.zeros(2)#np.array([r1*np.cos(t*10.0),r1*np.sin(t*10.0)]) 
                            actions.append(action)
                        elif agent_id == 2:
                            r2 = 1.5
                            action = np.zeros(2)#np.array([r2*np.cos(t*10.0),r2*np.sin(t*10.0)]) 
                            actions.append(action)
                        elif agent_id == 3:
                            r3 = 2.0
                            action = np.zeros(2)#np.array([r3*np.cos(t*10.0),r3*np.sin(t*10.0)]) 
                            actions.append(action)
                        elif agent_id == 4:
                            r3 = 2.0
                            action = np.zeros(2)#np.array([r3*np.cos(t*10.0),r3*np.sin(t*10.0)]) 
                            actions.append(action)
                        
                s_next, goal_next, r, done = self.env.step(actions)
                #print(s_next.shape,goal_next.shape)
                score += r[0] #all reward for each agent is the same
                self.buffer.store_episode(s[0], g[0], u[0], r[0], s_next[0], goal_next[0])
                s = s_next
                g = goal_next

                if self.buffer.current_size >= self.args.batch_size_im:
                    transitions = self.buffer.sample(self.args.batch_size_im)
                    self.agent.learn(transitions)

                self.noise = max(0.05, self.noise - 0.0000005)
                self.epsilon = max(0.05, self.noise - 0.0000005)

                if done[0]:
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
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)

        return sum(returns) / self.args.evaluate_episodes
