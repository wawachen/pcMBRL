import numpy as np
from os.path import dirname, join, abspath

from pyrep.envs.orca_bacterium_environment_demo_v0_MBRL_demo import Drone_Env

from os.path import dirname, join, abspath

# from pyrep.policies.dqn import DQNAgent
from pyrep.common.arguments_v0 import get_args
from pyrep.common.rollout_drone_navigation_demo_MBRL import Rollout


if __name__ == '__main__':
    # get the params
    args = get_args()
    num_agents = args.n_agents

    if args.field_size == 10:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')
    if args.field_size == 15:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_15x15.ttt')

    # create multiagent environment
    env = Drone_Env(args, env_name,num_agents)
 
    args.high_action = 1 
    args.load_buffer = False
    args.save_rate = 1000
    args.save_dir = "./" + args.scenario_name + "/orca_model_drone{}".format(args.n_agents)+'/'+'field_size{}'.format(args.field_size)
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)

    rollout = Rollout(args, env)
    # print(args.evaluate)
    if args.evaluate:
        rollout.evaluate()
    else:
        observations,actions,observations_next = rollout.run()
        
        fileName = "orca_demonstrationMBRL_steps50k"

        fileName += "_" + str(args.n_agents) + "agents"+"_"+"env"+str(args.field_size)

        fileName += ".npz"
    
        np.savez_compressed(fileName, acs=actions, obs=observations, obs_next=observations_next)
        print("finish saving file")
        # print('average reward sum: {0}'.format(np.mean(rewards_sum)))
    
    env.shutdown()









