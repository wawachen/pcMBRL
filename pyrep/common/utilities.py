import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np

def transpose_list(mylist):
    return list(map(list, zip(*mylist)))

def transpose_to_tensor(input_list):
    make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=0.5, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y

"""def main():
    torch.Tensor()
    print(onehot_from_logits())

if __name__=='__main__':
    main()"""

TPI = 2 * np.pi
PI32 = np.pi

def _product_difference(a, n):
    """
    Generates 2d matrix of differences between all pairs in the argument array
    i.e. y[i][j] = x[j]-x[i]  for all i,j where i≠j

    Args:
        a (np.array): 1d array of 32bit floats
        n (int): Number of entries in array that will form 1st index of result

    Returns:
        np.array: 2d Array of differences
    """
    m = a.shape[0]
    d = np.empty((n, m - 1), dtype=np.float32)
    for i in range(n):
        for j in range(i):
            d[i][j] = a[j] - a[i]
        for j in range(i + 1, m):
            d[i][j - 1] = a[j] - a[i]
    return d

def _torus_vectors(a, length, m):
    """
    Get pairs of vectors between two points, wrapping around the torus. This
    is done for all pairs of points in the argument array

    For example if x=1, y=2 and the torus is length 3, then there are 2 vectors
    from x to y: 1 and -2

    Args:
        a (np.array): 1d array of 32bit floats representing points
        length (float): Length of the torus/closed loop

    Returns:
        tuple(np.array, np.array): Tuple of 2d arrays containing the pairs
            of vectors for all pairs of points from the argument array
    """
    x = _product_difference(a, m)
    return x, np.sign(x) * (np.abs(x) - length)

def _shortest_vec(a, length, m):
    """
    Get the shortest vector between pairs of points taking into account
    wrapping around the torus

    Args:
        a (np.array): 1d array of 32bit floats representing the points
        length (float): Length of the torus/closed loop

    Returns:
        np.array: 2d array of shortest vectors between all pairs of points
    """
    a, b = _torus_vectors(a, length, m)
    return np.where(np.abs(a) < np.abs(b), a, b)


def _distances(xs, ys):
    """Convert x and y vector components to Euclidean distances"""
    return np.sqrt(np.power(xs, 2) + np.power(ys, 2))

def _distance_rewards1(d, proximity_threshold, distant_threshold, des):
    """
    Reward function based on distances between agents, for each agent
    the rewards are summed over contrib done for all pairs of points in the argument arrayutions from all other boids

    Args:
        d (np.array): 2d array of distances between pairs of boids
        proximity_threshold (float): Threshold distance at which boids are
            penalised for being too close

    Returns:
        np.array: 1d array of total rewards for each agent
    """
    distance_rewards = np.exp(-10 * d)
    distance_rewards_s = np.sort(distance_rewards)[:,:5]
    # distance_rewards = 1/d
    for i in range(d.shape[0]):
        for j in range(5):
            if d[i][j] < proximity_threshold or d[i][j] > distant_threshold or des[i] > 0.1: #or des[i] > 0.04:
                distance_rewards[i][j] = 0
            # if d[i][j] < proximity_threshold or d[i][j] > distant_threshold: #or des[i] > 0.04:
            #     if d[i][j] < proximity_threshold:
            #         distance_rewards[i][j] = 0
            #     else:
            #         distance_rewards[i][j] = 0
    return distance_rewards.sum(axis=1)
                                                                                    
def _distance_rewards(d, proximity_threshold, distant_threshold, des):
    """
    Reward function based on distances between agents, for each agent
    the rewards are summed over contrib done for all pairs of points in the argument arrayutions from all other boids

    Args:
        d (np.array): 2d array of distances between pairs of boids
        proximity_threshold (float): Threshold distance at which boids are
            penalised for being too close

    Returns:
        np.array: 1d array of total rewards for each agent
    """
    distance_rewards = np.exp(-10 * d)
    # distance_rewards = 1/d
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if d[i][j] < proximity_threshold or d[i][j] > distant_threshold or des[i] > 0.08:
                distance_rewards[i][j] = 0
            # if d[i][j] < proximity_threshold or d[i][j] > distant_threshold: #or des[i] > 0.04:
            #     if d[i][j] < proximity_threshold:
            #         distance_rewards[i][j] = -50
            #     else:
            #         distance_rewards[i][j] = 0
    return distance_rewards.sum(axis=1)

    # distance_rewards = 1/d
    # distance_rewards = np.zeros(d.shape[0])
    # d1 = np.sort(d)

    # for i in range(d.shape[0]):
    #     dnew = np.sum(d1[i,0:5])
    #     distance_rewards[i] = 1/dnew 
        

    # return distance_rewards  

def _separation_rewards(d,d_s):
    separation_r = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        d_min = d[i,:]<d_s
        num = np.sum(d_min)
        if num>0:
            separation_r[i] = -15
        else:
            separation_r[i] = 0
    return separation_r
                

def get_turning_angle(a,b):
    a1 = np.sqrt((a[0] * a[0]) + (a[1] * a[1]))
    b1 = np.sqrt((b[0] * b[0]) + (b[1] * b[1]))
    aXb = (a[0] * b[0]) + (a[1] * b[1])

    cos_ab = aXb/(a1*b1)
    angle_ab = math.acos(cos_ab)*(180.0/np.pi)
    return angle_ab

def _relative_headings(theta):
    """
    Get smallest angle between heading of all pairs of

    Args:
        theta (np.array): 1d array of 32bit floats representing agent headings
            in radians

    Returns:
        np.array: 2d array of 32bit floats representing relative headings
            for pairs of boids
    """
    return _shortest_vec(theta, TPI, theta.shape[0]) / PI32

# def _obstacle_penalties(ds,):
#         """
#         Return penalties for agent colliding with obstacles

#         Args:
#             ds (np.array): 2d array distances to obstacles for each agent

#         Returns:

#         """
#         # return -100 * np.any(ds < self.obstacle_radii, axis=1)
#         return -100 * np.any(ds < self.obstacle_radii, axis=1)

def take_along_axis(xs, sort_idx, dim=1):
    n1 = sort_idx.shape[1]
    n_agents = sort_idx.shape[0]
    xs1 = np.zeros([n_agents,n1])

    for i in range(n_agents):
        for j in range(n1):
            xs1[i,j] = xs[i,sort_idx[i,j]]
    
    return xs1