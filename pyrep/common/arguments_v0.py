import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="cooperative_navigation", help="name of the scenario script")
    parser.add_argument("--max-episodes", type=int, default=3000, help="maximum episode length")
    parser.add_argument("--max-episode-len", type=int, default=80, help="number of time steps")
    parser.add_argument("--seed", type=int, default=200, help="seed for random number generator")

    parser.add_argument("--load-type", type=str, default="three", help="different load types with different grasping points")
    parser.add_argument("--n-agents", type=int, default=3, help="number of agents")
    parser.add_argument("--field-size", type=int, default=10, help="field size for training")

    # train parameters
    parser.add_argument("--training", type=int, default=1, help="training mode (1 for training, 0 for evaluation)")
    parser.add_argument("--ntrain-iters", type=int, default=50, help="number of training iterations")
    parser.add_argument("--nrollouts-per-iter", type=int, default=5, help="number of rollouts per iteration")
    parser.add_argument("--ninit-rollouts", type=int, default=5, help="number of initial rollouts")
    parser.add_argument("--neval", type=int, default=1, help="number of evaluation rollouts")
    parser.add_argument("--task-hor", type=int, default=50, help="task horizon (automatically set based on n_agents if not specified)")
    parser.add_argument("--demo-data-file", type=str, default="", help="demonstration data file (automatically set based on n_agents if not specified)")
    parser.add_argument("--log-path", type=str, default="/home/xlab/MARL_transport/log_MBRL_model", help="log path for tensorboard")
    
    # MPC parameters
    parser.add_argument("--mpc-per", type=int, default=1, help="MPC per parameter")
    parser.add_argument("--mpc-prop-mode", type=str, default="TSinf", help="MPC propagation mode")
    parser.add_argument("--mpc-opt-mode", type=str, default="CEM", help="MPC optimization mode")
    parser.add_argument("--mpc-npart", type=int, default=20, help="MPC number of particles")
    parser.add_argument("--mpc-ign-var", default=False, action='store_true', help="MPC ignore variance")
    parser.add_argument("--mpc-plan-hor", type=int, default=5, help="MPC planning horizon")
    parser.add_argument("--mpc-num-nets", type=int, default=1, help="MPC number of networks")
    parser.add_argument("--mpc-epsilon", type=float, default=0.001, help="MPC epsilon")
    parser.add_argument("--mpc-alpha", type=float, default=0.25, help="MPC alpha")
    parser.add_argument("--mpc-epochs", type=int, default=25, help="MPC training epochs")
    parser.add_argument("--mpc-popsize", type=int, default=50, help="MPC population size")
    parser.add_argument("--mpc-max-iters", type=int, default=3, help="MPC maximum iterations")
    parser.add_argument("--mpc-num-elites", type=int, default=10, help="MPC number of elites")
    parser.add_argument("--mpc-load-model", default=False, action='store_true', help="MPC load model")
    
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--eval-episodes", type=int, default=20, help="number of episodes for evaluation")
    parser.add_argument("--evaluate", default=False, action='store_true', help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=10000, help="how often to evaluate model")
    args = parser.parse_args()

    return args
