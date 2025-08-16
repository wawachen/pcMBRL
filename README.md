### MARL Transport 快速使用说明

简洁指南：运行无人机（Drone）与 Turtlebot 的训练与推理脚本。

### 环境准备
- 安装依赖：
```bash
pip install -r requirements.txt
```

### Drone（无人机）
- 混合RL推理（推荐示例）
```bash
bash examples/train_RL_drones_whole_transport_hybrid.sh
```
  - 调用 `examples/RL_drone_transport_hybrid.py`，常用参数：`--n-agents`、`--field-size`、`--evaluate-episodes` 等。

- 离线 MBRL 训练（基于示例数据）
```bash
bash examples/train_offline_model_MBRL.sh
```
  - 调用 `examples/train_offline_model_MBRL.py`，针对 3/4/6 个智能体，使用仓库内的 `orca_demonstrationMBRL_steps50k_*agents_env10.npz`。

- 离线 MBRL 推理/评估
```bash
bash examples/evaluate_offline_model_MBRL.sh
```
  - 评估基于学习的动力学 + MPC 策略，默认日志目录：`/home/xlab/MARL_transport/log_MBRL_model`。

- 直接运行（示例，Drone MBRL）
```bash
python examples/train_offline_model_MBRL.py \
  --n-agents=3 \
  --field-size=10 \
  --training=1 \
  --log-path="/home/xlab/MARL_transport/log_MBRL_model" \
  --ntrain-iters=20 --nrollouts-per-iter=5 --ninit-rollouts=5 --neval=1 \
  --mpc-per=1 --mpc-prop-mode="TSinf" --mpc-opt-mode="CEM" \
  --mpc-npart=20 --mpc-plan-hor=10 --mpc-num-nets=1 \
  --mpc-epsilon=0.001 --mpc-alpha=0.25 --mpc-epochs=25 \
  --mpc-popsize=50 --mpc-max-iters=3 --mpc-num-elites=10
```
  - 切换到评估：将 `--training=0`，并添加 `--eval-episodes=20`。

### Turtlebot（小车）
- MBRL 训练
```bash
bash examples/train_turtlebot_MBRL.sh
```
  - 调用 `examples/train_turtlebot_MBRL.py`，针对 3/4 个智能体，使用 `turtlebot_demonstrationMBRL_steps50k_*agents_env10.npz`。

- MBRL 推理/评估
```bash
bash examples/evaluate_turtlebot_MBRL.sh
```
  - 日志默认保存到：`/home/xlab/MARL_transport/log_MBRL_model/evaluation_logs`。

- 直接运行（示例，Turtlebot MBRL）
```bash
python examples/train_turtlebot_MBRL.py \
  --n-agents=3 \
  --field-size=10 \
  --training=0 \
  --eval-episodes=20 \
  --log-path="/home/xlab/MARL_transport/log_MBRL_model" \
  --task-hor=40 \
  --mpc-per=1 --mpc-prop-mode="TSinf" --mpc-opt-mode="CEM" \
  --mpc-npart=20 --mpc-plan-hor=10 --mpc-num-nets=1 \
  --mpc-epsilon=0.001 --mpc-alpha=0.25 --mpc-epochs=25 \
  --mpc-popsize=50 --mpc-max-iters=3 --mpc-num-elites=10
```

### 备注
- 场景文件（`*.ttt`）位于 `examples/`，脚本会自动加载。
- `--n-agents` 可在 {3, 4, 6} 中切换，对应随附数据与配置。
- 日志与模型默认写入 `log_MBRL_model`，可通过 `--log-path` 修改。 