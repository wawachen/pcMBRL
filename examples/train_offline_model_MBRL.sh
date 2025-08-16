#!/bin/bash

# Train MBRL model with 4 agents
python examples/train_offline_model_MBRL.py \
--n-agents=3 \
--field-size=10 \
--load-type="three" \
--seed=342 \
--training=1 \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
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

# Train MBRL model with 4 agents
python examples/train_offline_model_MBRL.py \
--n-agents=4 \
--field-size=10 \
--load-type="four" \
--seed=342 \
--training=1 \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
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

# Train MBRL model with 4 agents
python examples/train_offline_model_MBRL.py \
--n-agents=6 \
--field-size=10 \
--load-type="six" \
--seed=342 \
--training=1 \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
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

# Train MBRL model with 4 agents
python examples/train_offline_model_MBRL.py \
--n-agents=3 \
--field-size=10 \
--load-type="three" \
--seed=342 \
--training=1 \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=15 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10 

# Train MBRL model with 4 agents
python examples/train_offline_model_MBRL.py \
--n-agents=4 \
--field-size=10 \
--load-type="four" \
--seed=342 \
--training=1 \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=15 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10 

# Train MBRL model with 4 agents
python examples/train_offline_model_MBRL.py \
--n-agents=6 \
--field-size=10 \
--load-type="six" \
--seed=342 \
--training=1 \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
--mpc-per=1 \
--mpc-prop-mode="TSinf" \
--mpc-opt-mode="CEM" \
--mpc-npart=20 \
--mpc-plan-hor=15 \
--mpc-num-nets=1 \
--mpc-epsilon=0.001 \
--mpc-alpha=0.25 \
--mpc-epochs=25 \
--mpc-popsize=50 \
--mpc-max-iters=3 \
--mpc-num-elites=10 

# Train MBRL model with 4 agents
python examples/train_offline_model_MBRL.py \
--n-agents=3 \
--field-size=10 \
--load-type="three" \
--seed=342 \
--training=1 \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
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
--mpc-num-elites=10 

# Train MBRL model with 4 agents
python examples/train_offline_model_MBRL.py \
--n-agents=4 \
--field-size=10 \
--load-type="four" \
--seed=342 \
--training=1 \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
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
--mpc-num-elites=10 

# Train MBRL model with 4 agents
python examples/train_offline_model_MBRL.py \
--n-agents=6 \
--field-size=10 \
--load-type="six" \
--seed=342 \
--training=1 \
--ntrain-iters=20 \
--nrollouts-per-iter=5 \
--ninit-rollouts=5 \
--neval=1 \
--log-path="/home/xlab/MARL_transport/log_MBRL_model" \
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
--mpc-num-elites=10 