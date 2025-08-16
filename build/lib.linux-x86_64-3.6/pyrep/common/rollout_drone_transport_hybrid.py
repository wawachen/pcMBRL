from pyrep.policies.maddpg_drone_att_demowp import MADDPG
from pyrep.policies.maddpg_agent import Agent  #modify
import os
import torch
import numpy as np
from tensorboardX import SummaryWriter

class Rollout:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.agents = [MADDPG(args,i) for i in range(args.n_agents)]
        self.agents_t = [Agent(i, args) for i in range(args.n_agents)]

    def evaluate(self,args, env):
        log_path = args.save_dir+"/log_drone_MPI_V0_demowp_evaluation_hybrid_transport"
        logger = SummaryWriter(logdir=log_path) # used for tensorboard
        returns = []

        for agent in self.agents:
            agent.prep_rollouts(device='cpu')

        t_c = 0

        for episode in range(args.evaluate_episodes):
            # reset the environment
            env.reset_world_t()
            s,s_t = env.reset_world()
            rewards = 0
           
            for time_step in range(args.evaluate_episode_len):
                actions = []
                actions_t = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action_evaluate(torch.tensor(s[agent_id], dtype=torch.float32).unsqueeze(0), 0.05, 0.05, args.use_gpu, logger, t_c) #sc[i], noise, epsilon,args.use_gpu,logger
                        actions.append(action[0,:])
                    
                    for agent_id, agent in enumerate(self.agents_t):
                        action_t = agent.select_action(s_t[agent_id], 0, 0)
                        actions_t.append(action_t)

                actions = np.array(actions)
                actions_t = np.array(actions_t)
                s_next, r, done, s_next_t, r_t, done_t = env.step(actions,actions_t)
                
                for j in range(args.n_agents):
                    logger.add_scalar('Agent%d/pos_x'%j, env.agents[j].agent.get_drone_position()[0], t_c)
                    logger.add_scalar('Agent%d/pos_y'%j, env.agents[j].agent.get_drone_position()[1], t_c)
                    logger.add_scalar('Agent%d/pos_z'%j, env.agents[j].agent.get_drone_position()[2], t_c)
                    logger.add_scalar('Agent%d/vel_x'%j, env.agents[j].agent.get_velocities()[0][0], t_c)
                    logger.add_scalar('Agent%d/vel_y'%j, env.agents[j].agent.get_velocities()[0][1], t_c)
                    logger.add_scalar('Agent%d/vel_z'%j, env.agents[j].agent.get_velocities()[0][2], t_c)
                    
                # logger.add_scalar('Load/pos_x', env.payload.get_position()[0],t_c)
                # logger.add_scalar('Load/pos_y', env.payload.get_position()[1],t_c)
                # logger.add_scalar('Load/pos_z', env.payload.get_position()[2],t_c)
                
                s = s_next
                s_t = s_next_t
                t_c += 1

        return sum(returns) / args.evaluate_episodes 
