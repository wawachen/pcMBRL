#!/usr/bin/env python 
from os.path import dirname, join, abspath
from pyrep.envs.turtlebot_environment import Turtle_Env
# from pyrep.policies.dqn import DQNAgent
from pyrep.common.arguments import get_args
from pyrep.common.rollout_navigation import Rollout

if __name__ == '__main__':
    # get the params
    args = get_args()

    env_name = join(dirname(abspath(__file__)), 'scene_turtlebot_navigation1.ttt')

    num_agents = 3
    # create multiagent environment
    env = Turtle_Env(env_name,num_agents)
 
    args.max_episodes = 8000 #8000
    args.max_episode_len = 60 #60
    args.n_agents = num_agents # agent number in a swarm
    args.evaluate_rate = 10000 
    args.evaluate = True #
    args.load_buffer = False
    args.evaluate_episode_len = 30
    args.evaluate_episodes = 20
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

