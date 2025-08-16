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

    args = parser.parse_args()

    return args
