from pyrep.policies.maddpg_agent import Agent
from pyrep.common.replay_buffer import Buffer
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
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_path = os.getcwd()+"/log_ddf"

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    # def wall_avoid(self,agent,action):
    #     xpos = agent.state.p_pos[0] + action[0]*self.env.world.dt 
    #     ypos = agent.state.p_pos[1] + action[1]*self.env.world.dt
    #     wall_dists = np.array([np.abs(1.0-ypos),np.abs(1.0+ypos),np.abs(1.0+xpos),np.abs(1.0-xpos)]) # rangefinder: forward, back, left, right
    #     # print(wall_dists)
    #     wall_a = wall_dists<0.04
    #     #print(len(entity_dis),len(wall_dists))
    #     return np.sum(wall_a)>0

    def run(self):
        returns = []
        logger = SummaryWriter(logdir=self.log_path) # used for tensorboard
        score = 0
        done_all = 0

        for time_step in range(self.args.time_steps):
            # reset the environment
            # print(time_step)
            if (time_step % self.episode_limit == 0) or done_all:
                print("episode%d"%int(time_step/self.episode_limit),":",score)
                logger.add_scalar('episode_rewards', score, int(time_step/self.episode_limit))
                # logger.add_scalar('network_loss', loss_sum, i_episode)
                s = self.env.reset_world()
                score = 0
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    # if self.wall_avoid(self.env.agents[agent_id],action):
                    #     action[0] = 0
                    #     action[1] = 0
                    u.append(action)
                    actions.append(action)

            s_next, r, done, info = self.env.stepPID(actions)
            if any(done):
                done_all = 1
            else:
                done_all = 0
            score += r[0] #all reward for each agent is the same
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)

            # if time_step > 0 and time_step % self.args.evaluate_rate == 0:
            #     returns.append(self.evaluate())
            #     plt.figure()
            #     plt.plot(range(len(returns)), returns)
            #     plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
            #     plt.ylabel('average returns')
            #     plt.savefig(self.save_path + '/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)
            np.save(self.save_path + '/returns.pkl', returns)
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
                s_next, r, done, info = self.env.stepPID(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)

        return sum(returns) / self.args.evaluate_episodes
