#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p /home/xlab/MARL_transport/log_MBRL_model/evaluation_logs

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Evaluating turtlebot MBRL with 3 agents..."
echo "Output will be saved to: /home/xlab/MARL_transport/log_MBRL_model/evaluation_logs/turtlebot_3agents_${TIMESTAMP}.log"

python examples/train_turtlebot_MBRL.py \
--n-agents=3 \
--field-size=10 \
--training=0 \
--seed=200 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--eval-episodes=20 \
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
--mpc-num-elites=10 \
--mpc-load-model 2>&1 | tee "/home/xlab/MARL_transport/log_MBRL_model/evaluation_logs/turtlebot_3agents_${TIMESTAMP}.log"

echo "Evaluating turtlebot MBRL with 4 agents..."
echo "Output will be saved to: /home/xlab/MARL_transport/log_MBRL_model/evaluation_logs/turtlebot_4agents_${TIMESTAMP}.log"

python examples/train_turtlebot_MBRL.py \
--n-agents=4 \
--field-size=10 \
--training=0 \
--seed=200 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--eval-episodes=20 \
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
--mpc-num-elites=10 \
--mpc-load-model 2>&1 | tee "/home/xlab/MARL_transport/log_MBRL_model/evaluation_logs/turtlebot_4agents_${TIMESTAMP}.log"

# echo "Evaluating turtlebot MBRL with 6 agents..."
# echo "Output will be saved to: /home/xlab/MARL_transport/log_MBRL_model/evaluation_logs/turtlebot_6agents_${TIMESTAMP}.log"
# 
# python examples/train_turtlebot_MBRL.py \
# --n-agents=6 \
# --field-size=10 \
# --training=0 \
# --seed=200 \
# --log-path="/home/xlab/MARL_transport/log_MBRL_model" \
# --eval-episodes=20 \
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
# --mpc-num-elites=10 \
# --mpc-load-model 2>&1 | tee "/home/xlab/MARL_transport/log_MBRL_model/evaluation_logs/turtlebot_6agents_${TIMESTAMP}.log"

echo "All turtlebot MBRL evaluation completed!"
echo "Log files saved in: /home/xlab/MARL_transport/log_MBRL_model/evaluation_logs/"
echo "Timestamp: ${TIMESTAMP}" 