import torch
from pyrep.networks.model_attention import Actor_att
from torch.utils.tensorboard import SummaryWriter
import argparse

# default `log_dir` is "runs" - we'll be more specific here
parser = argparse.ArgumentParser("multiagent environments")

# Environment
parser.add_argument("--scenario-name", type=str, default="cooperative_navigation", help="name of the scenario script")
parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
parser.add_argument("--time-steps", type=int, default=2000, help="number of time steps")

# parser.add_argument("--num-agents", type=int, default=20, help="number of agents")

# Core training parameters
parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
parser.add_argument("--lr-actor-im", type=float, default=1e-4, help="learning rate of actor")
parser.add_argument("--lr-critic-im", type=float, default=1e-3, help="learning rate of critic")
parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")

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
args.n_agents = 3

writer = SummaryWriter(log_dir='./log_VV')
net = Actor_att(args)
# obs = torch.tensor([[0.2,0.2,0.3,0.3,0.1,0.1,0.2,0.2,0.6,0.6,0.8,0.8,0.2,0.2]])
obs = torch.tensor([[0.2,0.2,0.3,0.3,0.1,0.1,0.2,0.2,0.6,0.6,0.8,0.8,0.2,0.2,0.3,0.3,0.1,0.1,0.2,0.2,0.6,0.6,0.8,0.8,0.2,0.2,0.3,0.3,0.1,0.1,0.2,0.2,0.6,0.6,0.8,0.8,0.1,0.1,0.2,0.2,0.3,0.3]])
writer.add_graph(net, obs)
writer.close()