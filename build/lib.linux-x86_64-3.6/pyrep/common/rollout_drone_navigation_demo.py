import os
from tensorboardX import SummaryWriter
import numpy as np

class Rollout:
    def __init__(self, args, env):
        self.args = args
        self.episode_limit = args.max_episode_len
        self.max_episodes = args.max_episodes
        self.restart_frequency = 500
        self.env = env
        self.log_path = args.save_dir +"/orca_log_drone"

        self.record_actions = []
        self.record_states = []
        self.record_states_next = []
        self.record_rewards = []
        self.record_rewardsum = []

    def run(self):
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
            succ_list = np.zeros(self.episode_limit)

            for t in range(self.episode_limit):
                # print(t)
                # start = time.time()
                s_next, u, r, done,succ = self.env.step_orca() 

                succ_list[t] = succ
                
                ep_states.append(s)
                ep_actions.append(u)
                ep_states_next.append(s_next)
                ep_rewards.append(r)

                score += r[0] #all reward for each agent is the same

                s = s_next

                # if np.any(done):
                #     break
                # if np.sum(succ_list)==10:
                #     break

                # if np.any(done):
                #     break
        
            print("collecting episode%d"%i,":",score)
            self.record_states.append(ep_states)
            self.record_actions.append(ep_actions)
            self.record_states_next.append(ep_states_next)
            self.record_rewards.append(ep_rewards)
            self.record_rewardsum.append(score)
            
        return self.record_states,self.record_actions,self.record_states_next,self.record_rewards,self.record_rewardsum
    
    def evaluate(self):
        # create the dict for store the model
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        logger = SummaryWriter(logdir=self.log_path) # used for tensorboard
        score = 0

        for i in range(self.args.evaluate_episodes):
            if ((i%self.restart_frequency)==0)and(i!=0):
                self.env.restart()
            s = self.env.reset_world()
            score = 0
            #for storage
            t_c = 0
           
            for t in range(self.args.evaluate_episode_len):
                # start = time.time()
                s_next, u, r, done = self.env.step_orca() 

                score += r[0] #all reward for each agent is the same
                for j in range(self.args.n_agents):
                    logger.add_scalar('Agent%d/pos_x'%j, s_next[j,2]*self.args.field_size, t_c)
                    logger.add_scalar('Agent%d/pos_y'%j, s_next[j,3]*self.args.field_size, t_c)
                    logger.add_scalar('Agent%d/vel_x'%j, s_next[j,0], t_c)
                    logger.add_scalar('Agent%d/vel_y'%j, s_next[j,1], t_c)
                s = s_next
                t_c +=1

                # if np.any(done):
                #     break

            logger.add_scalar('mean_episode_rewards', score, i)
            # logger.add_scalar('network_loss', loss_sum, i)
        
            print("episode%d"%i,":",score)
            
        logger.close()
        # return self.record_states,self.record_actions,self.record_states_next,self.record_rewards
