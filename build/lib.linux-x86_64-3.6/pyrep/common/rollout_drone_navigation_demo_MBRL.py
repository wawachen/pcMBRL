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
        max_steps = self.max_episodes  # Maximum number of steps to collect
        steps_collected = 0
        episode_count = 0

        # Initialize environment
        s = self.env.reset_world()
        episode_states = []
        episode_actions = []
        episode_states_next = []
        
        print(f"Starting data collection for {max_steps} steps...")

        while steps_collected < max_steps:
            # Check if we need to restart the environment periodically
            if (episode_count % self.restart_frequency == 0) and (episode_count != 0):
                self.env.restart()
                s = self.env.reset_world()
                print(f"Environment restarted at step {steps_collected}")

            # Collect one step of data
            s_next, u, done = self.env.step_orca()
            
            episode_states.append(s)
            episode_actions.append(u)
            episode_states_next.append(s_next)
            
            s = s_next
            steps_collected += 1
            
            # Check if episode is done or if we've reached episode limit
            if np.any(done):
                # Store episode data
                self.record_states.append(np.array(episode_states))
                self.record_actions.append(np.array(episode_actions))
                self.record_states_next.append(np.array(episode_states_next))
                
                episode_count += 1
                print(f"Episode {episode_count} completed at step {steps_collected}")
                
                # Reset for next episode
                s = self.env.reset_world()
                episode_states = []
                episode_actions = []
                episode_states_next = []
                
            # Print progress every 1000 steps
            if steps_collected % 1000 == 0:
                print(f"Collected {steps_collected}/{max_steps} steps")

        # Store any remaining data if the last episode wasn't complete
        if episode_states:
            self.record_states.append(np.array(episode_states))
            self.record_actions.append(np.array(episode_actions))
            self.record_states_next.append(np.array(episode_states_next))
            
        print(f"Data collection completed: {steps_collected} steps in {episode_count} episodes")
        return np.concatenate(self.record_states), np.concatenate(self.record_actions), np.concatenate(self.record_states_next)
    
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
