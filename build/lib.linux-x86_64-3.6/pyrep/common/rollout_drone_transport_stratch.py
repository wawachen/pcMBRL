from pyrep.policies.maddpg_drone_att_demowp import MADDPG
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class Rollout:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.agents = [MADDPG(args,i) for i in range(args.n_agents)]

    def evaluate(self,args, env):
        log_path = args.save_dir+"/log_drone_MPI_V0_demowp_evaluation_whole_transport"
        logger = SummaryWriter(logdir=log_path) # used for tensorboard
        returns = []

        for agent in self.agents:
            agent.prep_rollouts(device='cpu')

        t_c = 0

        for episode in range(args.evaluate_episodes):
            # reset the environment
            s = env.reset_world()
            rewards = 0
           
            for time_step in range(args.evaluate_episode_len):
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action_evaluate(torch.tensor(s[agent_id], dtype=torch.float32).unsqueeze(0), 0.05, 0.05, args.use_gpu, logger, t_c) #sc[i], noise, epsilon,args.use_gpu,logger
                        actions.append(action[0,:])
                actions = np.array(actions)
                s_next, r, done = env.step(actions)
                
                for j in range(args.n_agents):
                    logger.add_scalar('Agent%d/pos_x'%j, env.agents[j].agent.get_drone_position()[0], t_c)
                    logger.add_scalar('Agent%d/pos_y'%j, env.agents[j].agent.get_drone_position()[1], t_c)
                    logger.add_scalar('Agent%d/pos_z'%j, env.agents[j].agent.get_drone_position()[2], t_c)
                    logger.add_scalar('Agent%d/vel_x'%j, env.agents[j].agent.get_velocities()[0][0], t_c)
                    logger.add_scalar('Agent%d/vel_y'%j, env.agents[j].agent.get_velocities()[0][1], t_c)
                    logger.add_scalar('Agent%d/vel_z'%j, env.agents[j].agent.get_velocities()[0][2], t_c)
                    
                logger.add_scalar('Load/pos_x', env.payload.get_position()[0],t_c)
                logger.add_scalar('Load/pos_y', env.payload.get_position()[1],t_c)
                logger.add_scalar('Load/pos_z', env.payload.get_position()[2],t_c)
                
                s = s_next
                t_c += 1

        return sum(returns) / args.evaluate_episodes 
