# MARL_transport
[Jingyu Chen](https://www.researchgate.net/profile/Jingyu-Chen-20) <br>   
The University of Sheffield

[PPT](https://drive.google.com/file/d/1W4y5NYy9OUnf1OauCdDr6KfAIPNrzIG8/view?usp=drive_link) | [Paper](https://www.sciencedirect.com/science/article/pii/S0921889023001288)

This is the official repository for the paper: A Deep Multi-Agent Reinforcement Learning Framework for Autonomous Aerial
Navigation to Grasping Points on Loads

## Install Dependencies
The project is based on the simulator of Coppliasim. Thus, make sure you have installed the Coppliasim. <br>
We use the ORCA for demonstration learning. The C++ code for ORCA can be found in `MARL_transport/src`. The Cython can transform the C++ code
into the python code. To complie it, run  
```
python setup_rvo.py build_ext –-inplace

```
As we used the parallel training for demo-MADDPG, we need to install
```
python -m pip install mpi4py

```

## Main 
### Demonstration collection
To collect the demonstration data, we run 
```
./collect_RL_drones_demo.sh

```
The data will be save like `orca_demonstration_ep100_3agents_env10.npz`


### Run the code
In the paper, we proposed two algorithms, `learning from the demonstrated trajectories` and `behaviour cloning`. We compare them with the MADDPG, MAPPO and the behaviour swarm optimisation (BSO).<br>

To run the plain MADDPG,
```
./train_RL_drones_V0.sh

```
To run the attention-based MADDPG,
```
./train_RL_drones_V0_attention.sh
```

To run the demonstrated trajectories,
```
./train_RL_drones_demo.sh

```

To run the behaviour cloning,
```
./train_RL_drones_orca.sh
```

To run the MAPPO,
```
./train_RL_drones_MAPPO.sh
```
To run the curriculum learning,
```
./train_curriculum_wt_ddpg.sh
```

To run the whole transport from navigation to manipulation,
```
./train_RL_drones_whole_transport_hybrid.sh
```

