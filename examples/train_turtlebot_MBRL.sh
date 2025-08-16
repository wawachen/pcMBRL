#!/bin/bash

# Train turtlebot MBRL model with different numbers of agents
echo "Training turtlebot MBRL with 3 agents..."
python examples/train_turtlebot_MBRL.py \
--n-agents=3 \
--field-size=10 \
--training=1 \
--seed=200 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--task-hor=40 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=10 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10

python examples/train_turtlebot_MBRL.py \
--n-agents=3 \
--field-size=10 \
--training=1 \
--seed=100 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--task-hor=40 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=10 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10

python examples/train_turtlebot_MBRL.py \
--n-agents=3 \
--field-size=10 \
--training=1 \
--seed=0 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--task-hor=40 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=10 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10

python examples/train_turtlebot_MBRL.py \
--n-agents=3 \
--field-size=10 \
--training=1 \
--seed=300 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--task-hor=40 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=10 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10


python examples/train_turtlebot_MBRL.py \
--n-agents=3 \
--field-size=10 \
--training=1 \
--seed=400 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--task-hor=40 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=10 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10


echo "Training turtlebot MBRL with 4 agents..."
python examples/train_turtlebot_MBRL.py \
--n-agents=4 \
--field-size=10 \
--training=1 \
--seed=200 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--ntrain-iters=50 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--task-hor=50 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=10 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10

echo "Training turtlebot MBRL with 4 agents..."
python examples/train_turtlebot_MBRL.py \
--n-agents=4 \
--field-size=10 \
--training=1 \
--seed=300 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--ntrain-iters=50 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--task-hor=50 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=10 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10

echo "Training turtlebot MBRL with 4 agents..."
python examples/train_turtlebot_MBRL.py \
--n-agents=4 \
--field-size=10 \
--training=1 \
--seed=400 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--ntrain-iters=50 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--task-hor=50 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=10 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10

echo "Training turtlebot MBRL with 4 agents..."
python examples/train_turtlebot_MBRL.py \
--n-agents=4 \
--field-size=10 \
--training=1 \
--seed=100 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--ntrain-iters=50 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--task-hor=50 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=10 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10

echo "Training turtlebot MBRL with 4 agents..."
python examples/train_turtlebot_MBRL.py \
--n-agents=4 \
--field-size=10 \
--training=1 \
--seed=0 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--ntrain-iters=50 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--task-hor=50 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=10 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10

# echo "Training turtlebot MBRL with 6 agents..."
# python examples/train_turtlebot_MBRL.py \
# --n-agents=6 \
# --field-size=10 \
# --training=1 \
# --seed=200 \
# --log-path="/home/xlab/MARL_transport/log_MBRL_model" \
# --ntrain-iters=50 \
# --nrollouts-per-iter=5 \
# --ninit-rollouts=5 \
# --neval=1 \
# --task-hor=60 \
# --mpc-per=1 \
# --mpc-prop-mode="TSinf" \
# --mpc-opt-mode="CEM" \
# --mpc-npart=20 \
# --mpc-plan-hor=10 \
# --mpc-num-nets=1 \
# --mpc-epsilon=0.001 \
# --mpc-alpha=0.25 \
# --mpc-epochs=25 \
# --mpc-popsize=50 \
# --mpc-max-iters=3 \
# --mpc-num-elites=10

echo "All turtlebot MBRL training completed!" 