#!/bin/bash

# Evaluate trained MBRL model with 4 agents
echo "evaluate three drones."
python examples/train_offline_model_MBRL.py \
--n-agents=3 \
--field-size=10 \
--load-type="three" \
--training=0 \
--seed=200 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--eval-episodes=20 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=20 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10 \
--mpc-load-model 

echo "evaluate four drones."
python examples/train_offline_model_MBRL.py \
--n-agents=4 \
--field-size=10 \
--load-type="four" \
--training=0 \
--seed=200 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--eval-episodes=20 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=20 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10 \
--mpc-load-model 

echo "evaluate six drones."
python examples/train_offline_model_MBRL.py \
--n-agents=6 \
--field-size=10 \
--load-type="six" \
--training=0 \
--seed=200 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--eval-episodes=20 \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=20 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10 \
--mpc-load-model 