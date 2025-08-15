import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=3, help="number of agents")
    parser.add_argument("--field-size", type=int, default=10, help="field size for training")
    parser.add_argument("--load-type", type=str, default="three", help="different load types with different grasping points")
    parser.add_argument("--scenario-name", type=str, default="cooperative_navigation", help="name of the scenario script")
    parser.add_argument("--is-pc", default=False, action='store_true', help="signal for curriculum")

    parser.add_argument("--is-local-obs", default=False, action='store_true', help="whether to use local observation")

    # parser.add_argument("--env-id", help="Name of environment")
    # parser.add_argument("model_name", help="Name of directory to store " + "model/training contents")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=10000, type=int)
    parser.add_argument("--episode_length", default=60, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,help="Number of updates per update cycle")
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    return config