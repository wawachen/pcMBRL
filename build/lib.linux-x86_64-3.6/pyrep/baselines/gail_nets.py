from pyrep.baselines.mlp_policy import Policy
from pyrep.baselines.mlp_critic import Value
from pyrep.baselines.mlp_discriminator import Discriminator
from torch import nn
import torch
import numpy as np
from pyrep.baselines.common import estimate_advantages
from pyrep.baselines.torch_u import *
import math
import os 

device = "cpu"
dtype = torch.float64

def to_device(device, *args):
    return [x.to(device) for x in args]

class GAIL:
    def __init__(self, args, state_dim, action_dim):
        self.policy_net = Policy(state_dim, action_dim, log_std=args.log_std)
        self.value_net = Value(state_dim)
        self.discrim_net = Discriminator(state_dim + action_dim)
        self.discrim_criterion = nn.BCELoss()
        to_device(device, self.policy_net, self.value_net, self.discrim_net, self.discrim_criterion)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=args.learning_rate)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.learning_rate)

        # optimization epoch number and batch size for PPO
        self.optim_epochs = 10
        self.optim_batch_size = 64
        self.args = args

        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir) 
        
        self.model_path = self.args.save_dir 
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if self.args.evaluate:
            if os.path.exists(self.model_path + '/params.pkl'):
                self.initialise_networks(self.model_path+'/params.pkl')
                print('Agent successfully loaded actor_network: {}'.format(
                                                                          self.model_path + '/params.pkl'))

    def select_action(self,state):
        action = self.policy_net.select_action(state)[0].numpy()
        return action
    
    def select_action_eval(self,state):
        action = self.policy_net(state)[0][0].numpy()
        return action

    def update_params(self, batch, expert_traj, i_iter,logger):
        states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
        actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)

        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.args.gamma, self.args.tau, device)

        """update discriminator"""
        for _ in range(1):
            expert_state_actions = torch.from_numpy(expert_traj).to(dtype).to(device)
            g_o = self.discrim_net(torch.cat([states, actions], 1))
            e_o = self.discrim_net(expert_state_actions)
            self.optimizer_discrim.zero_grad()
            discrim_loss = self.discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
                self.discrim_criterion(e_o, zeros((expert_traj.shape[0], 1), device=device))
            discrim_loss.backward()
            self.optimizer_discrim.step()
        
        logger.add_scalar('train/loss/d_loss', discrim_loss.item(),i_iter)
        logger.add_scalar('train/reward/expert_r', e_o.mean().item(), i_iter)
        logger.add_scalar('train/reward/gen_r', g_o.mean().item(), i_iter)
        
        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
        for _ in range(self.optim_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).to(device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                """update critic"""
                for _ in range(1):
                    values_pred = self.value_net(states_b)
                    value_loss = (values_pred - returns_b).pow(2).mean()
                    # weight decay
                    for param in self.value_net.parameters():
                        value_loss += param.pow(2).sum() * self.args.l2_reg
                    self.optimizer_value.zero_grad()
                    value_loss.backward()
                    self.optimizer_value.step()

                """update policy"""
                log_probs = self.policy_net.get_log_prob(states_b, actions_b)
                ratio = torch.exp(log_probs - fixed_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - self.args.clip_epsilon, 1.0 + self.args.clip_epsilon) * advantages_b
                policy_surr = -torch.min(surr1, surr2).mean()
                self.optimizer_policy.zero_grad()
                policy_surr.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 40)
                self.optimizer_policy.step()
                
            logger.add_scalar('train/loss/c_loss',value_loss.item(),i_iter)
            logger.add_scalar('train/loss/policy_loss',policy_surr.item(),i_iter)

    def initialise_networks(self, path):
        
        checkpoint = torch.load(path) # load the torch data

        self.policy_net.load_state_dict(checkpoint['policy_params'])    # actor parameters
        self.value_net.load_state_dict(checkpoint['value_params'])    # critic parameters
        self.optimizer_policy.load_state_dict(checkpoint['policy_optim_params']) # actor optimiser state
        self.optimizer_value.load_state_dict(checkpoint['value_optim_params']) # critic optimiser state
        self.discrim_net.load_state_dict(checkpoint['discrim_params']) 
        self.optimizer_discrim.load_state_dict(checkpoint['discrim_optim_params'])
        
    def save_model(self, num_ep):
        num = str(num_ep)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        save_dict = {'policy_params' : self.policy_net.state_dict(),
                    'policy_optim_params': self.optimizer_policy.state_dict(),
                    'value_params': self.value_net.state_dict(),
                    'value_optim_params': self.optimizer_value.state_dict(),
                    'discrim_params': self.discrim_net.state_dict(),
                    'discrim_optim_params': self.optimizer_discrim.state_dict()
                    }

        torch.save(save_dict, self.model_path + '/' + num + '_params.pkl')
