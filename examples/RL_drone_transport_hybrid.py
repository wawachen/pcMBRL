#!/usr/bin/env python 
from os.path import dirname, join, abspath
from pyrep.envs.bacterium_transport_environment_hybird import Drone_Env
# from pyrep.policies.dqn import DQNAgent
from pyrep.common.arguments_v1 import get_args
from pyrep.common.rollout_drone_transport_hybrid import Rollout


if __name__ == '__main__':
    # get the params
    args = get_args()

    if args.field_size == 10:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10_transport.ttt')
    if args.field_size == 15:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_15x15.ttt')

    # create multiagent environment
    env = Drone_Env(args, env_name,args.n_agents)
 
    args.high_action = 1
    args.load_buffer = False
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    
    args.save_dir = "./" + args.scenario_name + "/model_drone{}_demowp".format(args.n_agents)+'/'+'field_size{}'.format(args.field_size)+'/'+ 'env{}'.format(0)

    args.save_dir_t = "./analysis/hybrid/model_turtle3_att1"
    args.scenario_name_t = "navigation3_att"
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    args.use_gpu = False

    action_shape = []        
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2) 

    ###########################
    args.obs_shape_t = [env.observation_space_t[i].shape[0] for i in range(args.n_agents)]  # observation space
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape_t = []        
    for content in env.action_space_t[:args.n_agents]:
        action_shape_t.append(content.shape[0])
    args.action_shape_t = action_shape_t[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape_t[0]==2) 
    ###########################

    rollout = Rollout(args, env)
    
    rollout.evaluate(args, env)

    env.shutdown()

