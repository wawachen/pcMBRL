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
    parser.add_argument("--load-type", type=str, default="three", help="different load types with different grasping points")
    parser.add_argument("--is-pc", default=False, action='store_true', help="signal for curriculum")

    parser.add_argument("--is-local-obs", default=False, action='store_true', help="whether to use local observation")

    parser.add_argument("--is-sensor-obs", default=False, action='store_true', help="whether to use sensor observation")
    parser.add_argument("--local-sight", type=float, default=3.0, help="local sight")
    parser.add_argument("--n-agents", type=int, default=3, help="number of agents")

    parser.add_argument("--field-size", type=int, default=10, help="field size for training")

    parser.add_argument("--env-size", type=int, default=2, help="parallel environments for training")

    parser.add_argument("--save_model_epoch", type=int, default=10, help="Intervals for saving model")
    # parser.add_argument("--load_model_path", type=str, default="../model_pkl/MAGAIL_Train_2020-05-01_18:09:33", help="Path for loading trained model")

    # Core training parameters

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=800, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", default=False, action='store_true', help="whether to evaluate the model")

    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                        help='gae (default: 3e-4)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                        help='clipping epsilon for PPO')
    parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                        help='number of threads for agent (default: 4)')
    parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size per PPO update (default: 2048)')
    parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size for evaluation (default: 2048)')
    parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                        help='maximal number of main iterations (default: 500)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--save-model-interval', type=int, default=10, metavar='N',
                        help="interval between saving model (default: 0, means don't save)")
    parser.add_argument('--gpu-index', type=int, default=0, metavar='N')

    args = parser.parse_args()

    return args
