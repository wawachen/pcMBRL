import numpy as np

class TurtlebotRollout:
    def __init__(self, args, env):
        self.args = args
        self.episode_limit = args.max_episode_len
        self.max_episodes = args.max_episodes
        self.restart_frequency = 500  # 参考drone的设置
        self.env = env
        self.log_path = args.save_dir + "/turtlebot_log"

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
        
        print(f"Starting turtlebot data collection for {max_steps} steps...")

        while steps_collected < max_steps:
            # Check if we need to restart the environment periodically
            if (episode_count % self.restart_frequency == 0) and (episode_count != 0):
                self.env.restart()
                s = self.env.reset_world()
                print(f"Environment restarted at step {steps_collected}")
                episode_count+=1

            # Collect one step of data using random actions
            s_next, actions, done = self.env.step()
            
            episode_states.append(s)
            episode_actions.append(actions)
            episode_states_next.append(s_next)
            
            s = s_next
            steps_collected += 1
            
            # Check if episode is done or if we've reached episode limit
            if np.any(done):
                self.record_states.append(np.array(episode_states))
                self.record_actions.append(np.array(episode_actions))
                self.record_states_next.append(np.array(episode_states_next))
                
                episode_count += 1
                print(f"Episode {episode_count} completed at step {steps_collected}")

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
            
        print(f"Turtlebot data collection completed: {steps_collected} steps in {episode_count} episodes")
        return np.concatenate(self.record_states), np.concatenate(self.record_actions), np.concatenate(self.record_states_next)