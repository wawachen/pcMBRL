import threading
import numpy as np
import pickle


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
                self.buffer['plane_%d' % i] = np.zeros([self.size, 9*4+1]) # max nine neighbours with neighbour number
                self.buffer['prefer_vel_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
                self.buffer['done_%d' % i] = np.empty([self.size])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next, planes,prefer_vels,done):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
                self.buffer['done_%d' % i][idxs] = done[i]
                
                for j in range(planes[i].shape[0]):
                    # print(planes[i][j][:].shape)
                    self.buffer['plane_%d'%i][idxs][j*4:j*4+4] = planes[i][j][:]
                self.buffer['plane_%d'%i][idxs][36] = planes[i].shape[0] # add neighbour number
                
                self.buffer['prefer_vel_%d'%i][idxs] = prefer_vels[i]

    def store_episode1(self, o, u, r, o_next,done):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
                self.buffer['done_%d' % i][idxs] = done[i]
    
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

