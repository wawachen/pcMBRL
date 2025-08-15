#!/usr/bin/env python 
from os.path import dirname, join, abspath
from pyrep.envs.turtlebot_environment import Turtle_Env
# from pyrep.policies.dqn import DQNAgent
# from pyrep.common.arguments import get_args
from pyrep.common.rollout_navigation import Rollout
import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="cooperative_navigation", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000, help="number of time steps")
    
    # parser.add_argument("--num-agents", type=int, default=20, help="number of agents")

    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--lr-actor-att", type=float, default=1e-2, help="learning rate of actor1")
    parser.add_argument("--lr-critic-att", type=float, default=1e-2, help="learning rate of critic1")
    parser.add_argument("--lr-actor-im", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic-im", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time") #256
    parser.add_argument("--demo-batch-size", type=int, default=1024, help="number of demo episodes to optimize at the same time") #256
    parser.add_argument("--batch-size-im", type=int, default=80, help="number of episodes to optimize at the same time")
    
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=800, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # get the params
    args = get_args()

    env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')

    num_agents = 3
    # create multiagent environment
    env = Turtle_Env(env_name,num_agents)
 
    args.max_episodes = 8000 #8000
    args.max_episode_len = 80 #60
    args.n_agents = num_agents # agent number in a swarm
    args.evaluate_rate = 10000 
    args.evaluate = True #
    args.load_buffer = False
    args.evaluate_episode_len = 40
    args.evaluate_episodes = 1
    args.save_rate = 1000
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    args.save_dir = "./model_turtle3_att"
    args.scenario_name = "navigation3_att"
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []        
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2) 

    rollout = Rollout(args, env)
    if args.evaluate:
        returns = rollout.evaluate()
        print('Average returns is', returns)
    else:
        rollout.run()
    
    env.shutdown()

