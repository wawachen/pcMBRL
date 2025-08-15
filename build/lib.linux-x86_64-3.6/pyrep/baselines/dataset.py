import numpy as np


class Dset(object):
    def __init__(self, inputs, labels, nobs, all_obs, rews, randomize, num_agents, nobs_flag=False):
        self.inputs = inputs.copy()
        self.labels = labels.copy()
        self.nobs_flag = nobs_flag
        if nobs_flag:
            self.nobs = nobs.copy()
        self.all_obs = all_obs.copy()
        self.rews = rews.copy()
        self.num_agents = num_agents
        assert len(self.inputs[0]) == len(self.labels[0])
        self.randomize = randomize
        self.num_pairs = len(inputs[0])
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            for k in range(self.num_agents):
                self.inputs[k] = self.inputs[k][idx, :]
                self.labels[k] = self.labels[k][idx, :]
                if self.nobs_flag:
                    self.nobs[k] = self.nobs[k][idx, :]
                self.rews[k] = self.rews[k][idx]
            self.all_obs = self.all_obs[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels, self.all_obs, self.rews
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs, labels, rews, nobs = [], [], [], []
        for k in range(self.num_agents):
            inputs.append(self.inputs[k][self.pointer:end, :])
            labels.append(self.labels[k][self.pointer:end, :])
            rews.append(self.rews[k][self.pointer:end])
            if self.nobs_flag:
                nobs.append(self.nobs[k][self.pointer:end, :])
        all_obs = self.all_obs[self.pointer:end, :]
        self.pointer = end
        if self.nobs_flag:
            return inputs, labels, nobs, all_obs, rews
        else:
            return inputs, labels, all_obs, rews

    def update(self, inputs, labels, nobs, all_obs, rews, decay_rate=0.9):
        idx = np.arange(self.num_pairs)
        np.random.shuffle(idx)
        l = int(self.num_pairs * decay_rate)
        # decay
        for k in range(self.num_agents):
            self.inputs[k] = self.inputs[k][idx[:l], :]
            self.labels[k] = self.labels[k][idx[:l], :]
            if self.nobs_flag:
                self.nobs[k] = self.nobs[k][idx[:l], :]
            self.rews[k] = self.rews[k][idx[:l]]
        self.all_obs = self.all_obs[idx[:l], :]
        # update
        for k in range(self.num_agents):
            self.inputs[k] = np.concatenate([self.inputs[k], inputs[k]], axis=0)
            self.labels[k] = np.concatenate([self.labels[k], labels[k]], axis=0)
            if self.nobs_flag:
                self.nobs[k] = np.concatenate([self.nobs[k], nobs[k]], axis=0)
            self.rews[k] = np.concatenate([self.rews[k], rews[k]], axis=0)
        self.all_obs = np.concatenate([self.all_obs, all_obs], axis=0)
        self.num_pairs = len(inputs[0])
        self.init_pointer()


class MADataSet(object):
    def __init__(self, expert_path, train_fraction=0.7, ret_threshold=None, traj_limitation=np.inf, randomize=True,
                 nobs_flag=False):

        """episode_batch:  self.demo_episode x timestep x n x keydim   
        """
        
        self.nobs_flag = nobs_flag
        traj_data = np.load(expert_path)
        action_episodes = traj_data['acs']
        obs_episodes = traj_data['obs']
        obs_next_episodes = traj_data['obs_next']
        reward_episodes = traj_data['r']

        assert(action_episodes.shape[0]==500)
        assert(action_episodes.shape[1]==25)

        num_agents = action_episodes.shape[2]
        if num_agents == 3:
            demo_timesteps = 12
        if num_agents == 4:
            demo_timesteps = 25
        if num_agents == 6:
            demo_timesteps = 35

        obs = []
        acs = []
        rets = []
        lens = []
        rews = []
        obs_next = []

        all_obs = []
        for k in range(num_agents):
            obs.append([])
            acs.append([])
            rews.append([])
            rets.append([])
            obs_next.append([])

        perm_index = np.random.permutation(traj_data['obs'].shape[0])

        for index in perm_index:
            if len(lens) >= traj_limitation:
                break
            for k in range(num_agents):
                obs[k].append(obs_episodes[index,:demo_timesteps,k,:])
                acs[k].append(action_episodes[index,:demo_timesteps,k,:])
                obs_next[k].append(obs_next_episodes[index,:demo_timesteps,k,:])
                rews[k].append(reward_episodes[index,:demo_timesteps,k])
                rets[k].append(np.sum(reward_episodes[index,:demo_timesteps,k]))
                
            lens.append(demo_timesteps)
            all_obs.append(np.concatenate([obs[0][0],obs[1][0],obs[2][0]],axis=1))

        # n_agents, episode, time_step, obs_dim
        # print("observation shape:", num_agents, action_episodes.shape[0], len(obs[0][0]), len(obs[0][0][0]))
        # print("action shape:", len(acs), len(acs[0]), len(acs[0][0]), len(acs[0][0][0]))
        # print("reward shape:", len(rews), len(rews[0]), len(rews[0][0]))
        # print("return shape:", len(rets), len(rets[0]))
        # episode, time_step, obs_dim_all
        # print("all observation shape:", len(all_obs), len(all_obs[0]), len(all_obs[0][0]))
        self.num_traj = action_episodes.shape[0]
        self.avg_ret = np.mean(np.sum(rets, axis=1) / len(rets[0]))
        self.avg_len = sum(lens) / len(lens)
        self.rets = np.array(rets)
        self.lens = np.array(lens)
        self.obs = obs
        self.acs = acs
        self.obs_next = obs_next
        self.rews = rews

        for k in range(num_agents):
            self.obs[k] = np.concatenate(self.obs[k]) # timestep, obs_dim
            self.acs[k] = np.concatenate(self.acs[k])
            self.rews[k] = np.concatenate(self.rews[k]) # timestep
            self.obs_next[k] = np.concatenate(self.obs_next[k])
        self.all_obs = np.concatenate(all_obs)

        # get next observation
        # self.obs_next = obs_next

        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)

        assert self.obs[0].shape[0] == self.acs[0].shape[0]

        # print(self.obs[1].shape)
        self.num_transition = self.obs[0].shape[0]
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.obs_next, self.all_obs, self.rews, self.randomize, num_agents,
                         nobs_flag=self.nobs_flag)
        # for behavior cloning
        self.train_set = Dset(self.obs, self.acs, self.obs_next, self.all_obs, self.rews, self.randomize, num_agents,
                              nobs_flag=self.nobs_flag)
        self.val_set = Dset(self.obs, self.acs, self.obs_next, self.all_obs, self.rews, self.randomize, num_agents,
                            nobs_flag=self.nobs_flag)
        self.log_info()

    def log_info(self):
        print("Total trajectories: %d" % self.num_traj)
        print("Total transitions: %d" % self.num_transition)
        print("Average episode length: %f" % self.avg_len)
        print("Average returns:", str(self.avg_ret))

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, ret_threshold, traj_limitation):
    dset = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation)
    a, b, c, d = dset.get_next_batch(64)
    print(a[0].shape, b[0].shape, c.shape, d[0].shape)
    # dset.plot()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str,
                        default="orca_demonstration_ep500_3agents_env10.npz")
    parser.add_argument("--ret_threshold", type=float, default=-9.1)
    parser.add_argument("--traj_limitation", type=int, default=200)
    args = parser.parse_args()
    test(args.expert_path, args.ret_threshold, args.traj_limitation)
