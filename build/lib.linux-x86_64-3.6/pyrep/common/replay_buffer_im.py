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
            # we only use one agent
            self.buffer['o'] = [None]*self.size #np.empty([self.size, 64,512,2])
            self.buffer['g'] = np.empty([self.size, 2])
            self.buffer['u'] = np.empty([self.size, 2])
            self.buffer['r'] = np.empty([self.size])
            self.buffer['o_next'] = [None]*self.size #np.empty([self.size, 64,512,2])
            self.buffer['g_next'] = np.empty([self.size, 2])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, g, u, r, o_next, g_next):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        with self.lock:
            self.buffer['o'][idxs] = o
            self.buffer['g'][idxs] = g
            self.buffer['u'][idxs] = u
            self.buffer['r'][idxs] = r
            self.buffer['o_next'][idxs] = o_next
            self.buffer['g_next'][idxs] = g_next
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            if key == 'o' or key == 'o_next':
                temp_buffer[key] = [self.buffer[key][i] for i in list(idx)]
            else:
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

