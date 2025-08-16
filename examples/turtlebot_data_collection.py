import numpy as np
from os.path import dirname, join, abspath
import os

from pyrep.envs.turtlebot_environment_demo import Turtle_Env
from pyrep.common.arguments_v0 import get_args
from pyrep.common.rollout_turtlebot_data_collection import TurtlebotRollout


if __name__ == '__main__':
    # get the params
    args = get_args()
    num_agents = args.n_agents

    # 根据field_size选择场景文件
    if args.field_size == 10:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')
    elif args.field_size == 15:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_15x15.ttt')
    else:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')

    # 创建turtlebot环境
    env = Turtle_Env(args, env_name, num_agents)
 
    args.save_dir = "./" + args.scenario_name + "/turtlebot_model" + str(args.n_agents) + "/field_size" + str(args.field_size)

    rollout = TurtlebotRollout(args, env)
    
    print("Starting data collection...")
    observations, actions, observations_next = rollout.run()
    
    if observations is not None and actions is not None and observations_next is not None:
        # 构建文件名 - 参考drone的命名方式
        fileName = "turtlebot_demonstrationMBRL_steps50k"
        fileName += "_" + str(args.n_agents) + "agents"
        fileName += "_env" + str(args.field_size)
        fileName += ".npz"
        
        # 保存数据
        print(f"Saving data to {fileName}...")
        np.savez_compressed(fileName, acs=actions, obs=observations, obs_next=observations_next)
        
        print("=== Data Collection Summary ===")
        print(f"Saved file: {fileName}")
        print(f"Observations shape: {observations.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Next observations shape: {observations_next.shape}")
        print(f"Total data points: {len(observations)}")
        print("Data collection completed successfully!")
        
        # 数据质量检查
        print("\n=== Data Quality Check ===")
        print(f"Observation range: [{np.min(observations):.3f}, {np.max(observations):.3f}]")
        print(f"Action range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
        print(f"Next observation range: [{np.min(observations_next):.3f}, {np.max(observations_next):.3f}]")
        
        # 检查是否有异常值
        obs_nan_count = np.sum(np.isnan(observations))
        action_nan_count = np.sum(np.isnan(actions))
        obs_next_nan_count = np.sum(np.isnan(observations_next))
        
        if obs_nan_count + action_nan_count + obs_next_nan_count > 0:
            print(f"Warning: Found NaN values - obs: {obs_nan_count}, actions: {action_nan_count}, obs_next: {obs_next_nan_count}")
        else:
            print("No NaN values detected - data quality good!")
                
    # 关闭环境
    env.shutdown()

    print("Program completed!") 