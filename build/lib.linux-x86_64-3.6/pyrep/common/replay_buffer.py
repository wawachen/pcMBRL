import threading
import numpy as np
import pickle
import pyrep.common.rank_based as rank_based


class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        if args.load_buffer:
            self.load_buffer()
            self.current_size = self.buffer['current_size']
        else:
            self.current_size = 0
            # create the buffer to store info
            self.buffer = dict()
            for i in range(self.args.n_agents):
                self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
                self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
                self.buffer['r_%d' % i] = np.empty([self.size])
                self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
    
    def store_batch_episodes(self, episode_batch):
        """episode_batch:  self.demo_episode x timestep x n x keydim   
        """
        action_episodes = episode_batch['acs']
        action_episodes_batch = np.swapaxes(action_episodes,1,2)
        obs_episodes = episode_batch['obs']
        obs_episodes_batch = np.swapaxes(obs_episodes,1,2)
        batch_size = action_episodes.shape[1]
        assert(batch_size==60)
        assert(action_episodes.shape[0]==100)

        for j in range(action_episodes.shape[0]):
            with self.lock:
                idxs = self._get_storage_idx(batch_size)

                for i in range(self.args.n_agents):
                    self.buffer['o_%d' % i][idxs] = obs_episodes_batch[j,i,:]
                    self.buffer['u_%d' % i][idxs] = action_episodes_batch[j,i,:]

    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]    
        return temp_buffer

    def _get_storage_idx(self, inc=None): 
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def save_buffer(self):
        self.buffer['current_size'] = self.current_size
        with open('buffer.pickle', 'wb') as f:
            pickle.dump(self.buffer, f)
            print("buffer has been saved!!")

    def load_buffer(self):
        with open('buffer.pickle') as f:
            self.buffer = pickle.load(f)
            print("buffer has been loaded!!")


# class Pri_ReplayBuffer(object):

#   def __init__(self, args):

#     self.buffer_size = args.buffer_size
#     self.num_experiences = 0
#     self.p_n = 500 #partion number
#     self.learn_start = args.buffer_size/self.p_n
#     #self.buffer = deque()
#     conf = {'size': args.buffer_size,
#             'learn_start': self.learn_start,
#             'partition_num': self.p_n,
#             'total_step': args.max_episodes*20,
#             'batch_size': args.batch_size}
#     self.replay_memory = rank_based.Experience(conf)
  

#   def getBatch(self, batch_size):
#     # random draw N
#     #return random.sample(self.buffer, batch_size)
#     #none*[(traj)] 
#     batch, w, e_id = self.replay_memory.sample(self.num_experiences)
#     self.e_id=e_id
#     self.w_id=w
     
#     return batch, self.w_id, self.e_id


#   def size(self):
#     return self.buffer_size

#   def add(self, state, action, reward, next_state):#add(self, state, next_state, action, reward, done):
#     #new_experience = (state, next_action, action, reward, done)#(state, action, reward, next_state, done)
#     #[n,sdim] [n,adim]
#     self.replay_memory.store((state, action, reward, next_state))
#     #if self.num_experiences < self.buffer_size:
#     #  self.buffer.append(new_experience)
#     self.num_experiences += 1
#     #else:
#     #  self.buffer.popleft()
#     #  self.buffer.append(new_experience)

#   def count(self):
#     # if buffer is full, return buffer size
#     # otherwise, return experience counter
#     return self.num_experiences

#   #def erase(self):
#   #  self.buffer = deque()
#   #  self.num_experiences = 0
#   def rebalance(self):
#     self.replay_memory.rebalance()

#   def update_priority(self, indices, delta):
#     self.replay_memory.update_priority(indices, delta)

