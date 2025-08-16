from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as stats
from scipy.io import savemat


class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class CEMOptimizer(Optimizer):

    def __init__(self, sol_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.cost_function = cost_function
        self.record_i = 0

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        self.record_i = 0

    def obtain_solution(self, init_mean, init_var, goal):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, init_var, 0
        # truncate in [mu-2sigma, mu+2sigma] u = 0 sigma = 1
        #mean: [act_dim*plan_hor,]
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            # print("wawa", np.sqrt(constrained_var))

            #constrained and mean which is [sol_dim] will broadcast into [popsize, sol_dim]
            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)

            #costs [popsize,] store 15*[400,2] x z 
            costs,store_s = self.cost_function(samples, goal)

            #0719
            cost = costs[np.argsort(costs)][:self.num_elites]
            #[num_elites,sol_dim] 
            #np.argsort returns the index sorting from small to large
            elites = samples[np.argsort(costs)][:self.num_elites]
        
            # [soldim]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1
        
        ####visualisation
        self.record_i+=1
        savemat('/home/xlab/MARL_transport/data/traj/mpc_visualisation{0}.mat'.format(self.record_i), mdict={'arr': store_s})
        savemat('/home/xlab/MARL_transport/data/traj/mpc_costs{0}.mat'.format(self.record_i), mdict={'arr': costs})

        store_top_states = [] #[15,5,2]
        store_bad_states = []
        for i in range(len(store_s)):
            store_top_states.append(store_s[i][np.argsort(costs)[:5],:])
            store_bad_states.append(store_s[i][np.argsort(costs)[-5:],:])

        return mean,elites,np.array(store_top_states),np.array(store_bad_states),cost