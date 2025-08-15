import torch
import numpy as np
from pyrep.policies.MPC import MPC 
from pyrep.envs.Agent import Agent
from tqdm import trange
from tensorboardX import SummaryWriter
from time import localtime, strftime
import os
from pyrep.envs.turtlebot_environment import Turtle_Env
from pyrep.common.arguments_v0 import get_args
from os.path import dirname, join, abspath
from dotmap import DotMap

if __name__=="__main__":
    # get the params
    args = get_args()
    
    # Use args parameters for turtlebot
    n_agent = args.n_agents
    training = args.training
    num_agents = n_agent
    
    # Adjust task horizon for turtlebot (generally shorter than drones)
    if n_agent == 3:
        args.task_hor = 40
    elif n_agent == 4:
        args.task_hor = 50
    elif n_agent == 6:
        args.task_hor = 60
    
    # Set demo data file for turtlebot if not specified
    args.demo_data_file = 'turtlebot_demonstrationMBRL_steps50k_{0}agents_env10.npz'.format(n_agent)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use turtlebot simulation environment
    env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')  # Reuse the same scene
   
    env = Turtle_Env(args, env_name, num_agents)

    # For MPC - use args parameters adjusted for turtlebot
    params = DotMap()
    params.per = args.mpc_per
    params.prop_mode = args.mpc_prop_mode
    params.opt_mode = args.mpc_opt_mode
    params.npart = args.mpc_npart
    params.ign_var = args.mpc_ign_var
    params.plan_hor = args.mpc_plan_hor
    params.num_nets = args.mpc_num_nets
    params.epsilon = args.mpc_epsilon
    params.alpha = args.mpc_alpha
    params.epochs = args.mpc_epochs
    params.model_3d_in = n_agent*(6+4*(n_agent-1))+2*n_agent   # obs + action
    params.model_3d_out = n_agent*(6+4*(n_agent-1))   # next obs
    params.popsize = args.mpc_popsize
    params.max_iters = args.mpc_max_iters
    params.num_elites = args.mpc_num_elites
    params.load_model = (not training)
    params.robot = "turtlebot"

    policy = MPC(params, env)
    
    agent = Agent(env)
    
    # Use args parameters for training
    ntrain_iters = args.ntrain_iters
    nrollouts_per_iter = args.nrollouts_per_iter
    ninit_rollouts = args.ninit_rollouts
    neval = args.neval
    task_hor = args.task_hor

    if training:
        log_path = os.path.join(args.log_path, strftime("%Y-%m-%d--%H:%M:%S", localtime()))
    else:
        log_path = os.path.join(args.log_path)

    os.makedirs(log_path, exist_ok=True)
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    if training:

        demoDataFile = args.demo_data_file
        demoData = np.load(demoDataFile, allow_pickle=True)
        
        """episode_batch:  timestep x keydim   
        """
        action_episodes = demoData['acs']
        obs_episodes = demoData['obs']
        obs_next_episodes = demoData['obs_next']

        # 观察数据格式转换 - 针对2D数组 [timestep, keydim]
        def convert_turtlebot_obs_format(obs_data, num_agents):
            """
            将demoData的观察格式转换为环境兼容的格式
            
            输入: obs_data shape [timestep, keydim]
            [all_pos, all_vel, all_ori]
            其中: all_pos = [agent0_pos, agent1_pos, ..., agentN_pos] (每个位置2D)
                 all_vel = [agent0_vel, agent1_vel, ..., agentN_vel] (每个速度2D)  
                 all_ori = [agent0_ori, agent1_ori, ..., agentN_ori] (每个朝向3D)
            
            输出: 
            假设demoData格式: 按智能体分组 [agent0_all, agent1_all, ...]
            """
            timesteps, obs_dim = obs_data.shape
            obs_per_agent = obs_dim // num_agents
            
            # 重新组织观察数据
            converted_obs = []
            
            for t in range(timesteps):
                # 按特征类型收集数据
                new_obs = []

                pos_all = obs_data[t, :num_agents*2]
                vel_all = obs_data[t, num_agents*2:num_agents*4]
                ori_all = obs_data[t, num_agents*4:num_agents*7]

                for i in range(num_agents):

                    # 假设每个智能体的观察: [pos_x, pos_y, vel_x, vel_y, ori_x, ori_y, ori_z]
                    agent_pos = pos_all[i*2:i*2+2]
                    agent_vel = vel_all[i*2:i*2+2]
                    agent_ori = ori_all[i*3:i*3+3]
                    
                    # 提取位置 (前2维)
                    new_obs.append(agent_pos)
                    new_obs.append(agent_vel)
                    new_obs.append(np.array([np.sin(agent_ori[2]), np.cos(agent_ori[2])]))

                    for j in range(num_agents):
                        if j == i:
                            continue
                        else:
                            other_pos = pos_all[j*2:j*2+2] - agent_pos
                            other_ori = ori_all[j*3+2]

                            new_obs.append(other_pos)
                            new_obs.append(np.array([np.sin(other_ori-agent_ori[2]),np.cos(other_ori-agent_ori[2])]))
                
                converted_obs.append(np.concatenate(new_obs))
            
            return np.array(converted_obs)
        
        print(f"Original obs shape: {obs_episodes.shape}")
        print(f"Converting obs format for {num_agents} turtlebot agents...")
        
        # 转换观察数据格式
        obs_episodes = convert_turtlebot_obs_format(obs_episodes, num_agents)
        obs_next_episodes = convert_turtlebot_obs_format(obs_next_episodes, num_agents)
        
        print(f"Converted obs shape: {obs_episodes.shape}")

        # Perform initial rollouts
        samples = []

        train_obs = obs_episodes.reshape(-1, obs_episodes.shape[-1]) 
        train_acs = action_episodes.reshape(-1, action_episodes.shape[-1])
        train_obs_next = obs_next_episodes.reshape(-1, obs_next_episodes.shape[-1])

        #samples [episode, steps,n]
        policy.train(train_obs, train_acs, train_obs_next, logger)

        # Training loop
        for i in trange(ntrain_iters):
            print("####################################################################")
            print("Starting training iteration %d." % (i + 1))

            samples = []

            #horizon, policy, wind_test_type, adapt_size=None, log_data=None, data_path=None
            #MBRL is baseline no need to log data in agent sampling
            for j in range(max(neval, nrollouts_per_iter)):
                samples.append(
                        agent.sample(task_hor, policy)
                    )
            # print("Rewards obtained:", np.mean([sample["reward_sum"] for sample in samples[:]]))
            logger.add_scalar('Reward', np.mean([sample["reward_sum"] for sample in samples[:]]), i)
            samples = samples[:nrollouts_per_iter]

            if i < ntrain_iters - 1:
                #add new samples into the whole dataset and train the whole dataset
                policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["obs_next"] for sample in samples],
                    logger
                )

        logger.close()
    else:
        ###########################################################################
        # Evaluation mode for turtlebot
        returns = []
        time_f = []
        s_r = []
        deviation = []
        a_vel_l = []
        a_vel_a = []

        t_c = 0
        tt = 0
        t_c_ex = 0
        
        os.makedirs('data/turtlebot_traj', exist_ok=True)

        for episode in range(args.eval_episodes):
            # reset the environment
            s = env.reset_world()
            rewards = 0
            succ_list = np.zeros(task_hor)
            pos_x = []
            pos_y = []
            
            vel_mag1_a = []
            vel_mag1_l = []
            policy.reset()
            
            for time_step in range(task_hor):
                vel_mag_angular = []
                vel_mag_linear = []
                actions = []
                with torch.no_grad():
                    action, act_l, store_top_s, store_bad_s, costs = policy.act(s, time_step, env.goals[:,:2])
                
                s_next, reward, done, succ = env.step(action)
                succ_list[time_step] = succ

                len_per_agent = len(s_next)//args.n_agents
                for j in range(args.n_agents):
                    pos_x.append(s_next[j*len_per_agent]*env.field_size)
                    pos_y.append(s_next[j*len_per_agent+1]*env.field_size)    
                    vel_mag_linear.append(s_next[j*len_per_agent+2])
                    vel_mag_angular.append(s_next[j*len_per_agent+3])

                vel_mag1_a.append(np.sum(np.array(vel_mag_angular))/len(vel_mag_angular))   
                vel_mag1_l.append(np.sum(np.array(vel_mag_linear))/len(vel_mag_linear))
                
                rewards += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                s = s_next
                t_c += 1
                if np.any(done):
                    break
                if np.sum(succ_list) >= 10:  # Success criteria
                    break
            
            # Save trajectory for turtlebot
            pos_x_arr = np.array(pos_x).reshape(-1, args.n_agents)
            pos_y_arr = np.array(pos_y).reshape(-1, args.n_agents)
            # np.savez(f'data/turtlebot_traj/n={args.n_agents}episode_{episode+1}.npz', pos_x=pos_x_arr, pos_y=pos_y_arr)

            returns.append(rewards)
            
            if np.any(succ_list):
                time_f.append(t_c-t_c_ex)
                a_vel_a.append(np.sum(np.array(vel_mag1_a))/len(vel_mag1_a))
                a_vel_l.append(np.sum(np.array(vel_mag1_l))/len(vel_mag1_l))
            t_c_ex = t_c
            s_r.append(np.any(succ_list))

            # Calculate deviation for turtlebot (ground-based navigation)
            if np.any(succ_list):
                pos_x_end = pos_x[-args.n_agents*10:]
                pos_y_end = pos_y[-args.n_agents*10:]
                
                # Define goal positions for turtlebot (adjusted for ground navigation)
                if args.n_agents == 3:
                    goals = np.array([[3.0, 0], [-3.0, 0], [0, 0]])
                elif args.n_agents == 4:
                    goals = np.array([[2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]])
                elif args.n_agents == 6:
                    goals = np.array([[3.0, -3.0], [-3.0, -3.0], [1.5, 0], [-1.5, 0], [3.0, 3.0], [-3.0, 3.0]])
                
                dev_all = []
                for m in range(min(10, len(pos_x_end)//args.n_agents)):
                    d_e_t = 0
                    for i in range(args.n_agents):
                        d_e = np.zeros(args.n_agents)
                        for j in range(args.n_agents):
                            if m * args.n_agents + j < len(pos_x_end):
                                d_e[j] = np.sqrt(np.sum((goals[i,:] - np.array([
                                    pos_x_end[m * args.n_agents + j],
                                    pos_y_end[m * args.n_agents + j]
                                ]))**2))
                        d_e_t += np.min(d_e)
                    d_e_ta = d_e_t / args.n_agents
                    dev_all.append(d_e_ta)
                if dev_all:
                    deviation.append(np.sum(np.array(dev_all))/len(dev_all))

            print('Returns is', rewards)
            print("time_steps", time_step)
            
        print("=== Turtlebot MBRL Evaluation Results ===")
        if time_f:
            print("Finished time: ", np.sum(np.array(time_f))/len(time_f), ', ', np.std(np.array(time_f)))
        if a_vel_l:
            print('Average linear speed: ', np.sum(np.array(a_vel_l))/len(a_vel_l), ', ', np.std(np.array(a_vel_l)))
        if a_vel_a:
            print('Average angular speed: ', np.sum(np.array(a_vel_a))/len(a_vel_a), ', ', np.std(np.array(a_vel_a)))
        print("Success rate: ", np.sum(np.array(s_r))/len(s_r))
        if deviation:
            print("Deviation: ", np.sum(np.array(deviation))/len(deviation), ', ', np.std(np.array(deviation)))
        print("Total rewards", np.sum(np.array(returns)))
        logger.close()

    # Shutdown environment
    env.shutdown() 