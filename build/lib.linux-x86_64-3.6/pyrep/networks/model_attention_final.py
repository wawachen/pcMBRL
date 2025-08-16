from typing import DefaultDict
import torch 
import math 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
import os


#--------------drone--------------------------------

class Actor_att1(nn.Module):
    def __init__(self, args, with_action = False):
        super(Actor_att1,self).__init__()
        self.num_test = 32 // 2
        self.num_units = 32
        self.n_good = args.n_agents
        self.n_food = args.n_agents
        self.num_outputs = self.num_test
        self.with_action = with_action
        self.self_size = 4
        self.other_size = 4
        self.food_size = 2

        self.en_encoder = nn.Sequential()
        if self.with_action:
            self.en_encoder.add_module('en_coder1',nn.Linear(self.self_size+2, self.num_units))
        else:
            self.en_encoder.add_module('en_coder1',nn.Linear(self.self_size, self.num_units))
        self.en_encoder.add_module('en_af1',nn.ReLU())
        self.en_encoder.add_module('en_coder2',nn.Linear(self.num_units, self.num_test))
        self.en_encoder.add_module('en_af2',nn.ReLU())

        self.oa_encoder = nn.Sequential()
        self.oa_encoder.add_module('oa_coder1',nn.Linear(self.other_size, self.num_units))
        self.oa_encoder.add_module('oa_af1',nn.ReLU())
        self.oa_encoder.add_module('oa_coder2',nn.Linear(self.num_units, self.num_test))
        self.oa_encoder.add_module('oa_af2',nn.ReLU())
        self.oa_attention = nn.Sequential()
        self.oa_attention.add_module('oa_bn', nn.LayerNorm(
                                            self.num_test))
        self.oa_attention.add_module('oa_bf1',nn.ReLU())

        self.goal_encoder = nn.Sequential()
        self.goal_encoder.add_module('goal_coder1',nn.Linear(self.food_size, self.num_units))
        self.goal_encoder.add_module('goal_af1',nn.ReLU())
        self.goal_encoder.add_module('goal_coder2',nn.Linear(self.num_units, self.num_test))
        self.goal_encoder.add_module('goal_af2',nn.ReLU())
        self.goal_attention = nn.Sequential()
        self.goal_attention.add_module('goal_bn', nn.LayerNorm(
                                            self.num_test))
        self.goal_attention.add_module('goal_bf1',nn.ReLU())


        self.merge_layer = nn.Sequential()
        if self.with_action:
            self.merge_layer.add_module('merge_l1',nn.Linear(self.num_test*3,self.num_units))
            self.merge_layer.add_module('merge_f1',nn.ReLU())
            self.merge_layer.add_module('merge_l2',nn.Linear(self.num_units,self.num_units))
            self.merge_layer.add_module('merge_f2',nn.ReLU())
            self.merge_layer.add_module('merge_l3',nn.Linear(self.num_units,self.num_outputs))
        else:
            self.merge_layer.add_module('merge_l1',nn.Linear(self.num_test*3,self.num_units))
            self.merge_layer.add_module('merge_f1', nn.LeakyReLU())
            self.merge_layer.add_module('merge_l2',nn.Linear(self.num_units,self.num_units))
            self.merge_layer.add_module('merge_f2', nn.LeakyReLU())
            self.merge_layer.add_module('merge_l3', nn.Linear(self.num_units,2))
            self.merge_layer.add_module('merge_f3', nn.Tanh())
            

    def forward(self, s_input, a_input = None, logger=None, agent_id=None, iter_n=None):
        #self mlp
        # print(s_input.shape) #[None, sdim]
        
        self_in = s_input[:,:4] 
        
        if self.with_action:
            #obs_self+ food+other_obs+self_action
            self_action = a_input[:,:]
            # print(self_action)
            self_in = torch.cat([self_in, self_action], dim=1) #[None, 6]
            # print(self_in.shape) 

        # print("self",self_in.shape)
        self_out = self.en_encoder(self_in) #[None,self.num_test]

        #other mlp
        other_good_in = s_input[:, 4:4+(self.n_good-1)*4]
        other_good_ins = []
        for i in range(self.n_good-1):
            temp1 = other_good_in[:, i*2:2+i*2] #[None,2] v  v+p 
            temp2 = other_good_in[:, (self.n_good-1)*2+i*2:(self.n_good-1)*2+i*2+2] #p
            temp = torch.cat([temp1,temp2],dim=1) #[none,4]
            assert(temp.shape[1]==4)
            other_good_ins.append(temp)

        other_good_ins = torch.stack(other_good_ins, dim=0) #[n_good-1,None,4]
        # print(other_good_ins.shape)
        
        other_outs = self.oa_encoder(other_good_ins) #[n_good-1,None,self.num_test]
        other_good_out = other_outs.permute(1, 2, 0) #[None,self.num_test,n_good-1]

        other_good_out_attn = F.softmax(torch.matmul(self_out.unsqueeze(1), other_good_out)/math.sqrt(self.num_test),dim=2) # [None,1,self.num_test]x[None,self.num_test,n_good-1]-> [None,1,n_other]

        if logger is not None:
            # print(other_good_out_attn.shape)
            logger.add_scalars('agent%i/actor_attention/other_agents' % agent_id,
                                   dict(('attention_weight%i_' % h_i, other_good_out_attn[0,0,h_i]) for h_i 
                                        in range(self.n_good-1)),
                                   iter_n)
        # print("attn:", other_good_out_attn.shape)
        # [None,1,n_other]x[None,n_other,self.num_test]->[None, 1,self.num_test]->[None,self.num_test]
        other_good_out = torch.matmul(other_good_out_attn, other_good_out.permute(0,2,1)).squeeze(1)
        
        other_good_out = self.oa_attention(other_good_out) #[None,self.num_test]
        # print(other_good_out.shape)

        #food mlp
        food_input = s_input[:,-self.n_food*2:] #[None,n_food*2]

        food_input = torch.split(food_input, 2, dim=1) #split size = 2 tuple()
        # print(len(food_input))
        food_input = torch.stack(list(food_input), dim=0) #[n_food,None,2]
        
        food_outs = self.goal_encoder(food_input) #[n_food,None, self.num_test]

        food_out = food_outs.permute(1, 2, 0) #[None,self.num_test,n_food]
        # print(torch.matmul(self_out.unsqueeze(1), food_out).shape)
        food_out_attn = F.softmax(torch.matmul(self_out.unsqueeze(1), food_out)/math.sqrt(self.num_test),dim=2) #[None, 1, self.num_test] x [None,self.num_test,n_food]->[None,1,n_food]

        if logger is not None:
            logger.add_scalars('agent%i/actor_attention/goals' % agent_id,
                                   dict(('attention_weight%i_' % h_i, food_out_attn[0,0,h_i]) for h_i 
                                        in range(self.n_good)),
                                   iter_n)

        food_out = torch.matmul(food_out_attn, food_out.permute(0,2,1)).squeeze(1)  #[None,self.num_test]
        food_out = self.goal_attention(food_out) 
        # print(food_out.shape)

        input_merge = torch.cat([self_out, food_out, other_good_out], 1) #[None,self.num_test*3]
        assert(input_merge.shape[1]==self.num_test*3)
        # print(input_merge.shape)

        out = self.merge_layer(input_merge)

        return out
        

class Critic_att1(nn.Module):
    def __init__(self, args, index):
        super(Critic_att1,self).__init__()
        self.num_units = 32
        self.num_test = self.num_units // 2
        self.n_good = args.n_agents
        self.n_food = args.n_agents
        self.num_outputs = 1
        self.index = index

        self.layer_actor1 = nn.Linear(self.num_test, self.num_test)
        self.layer_actor2 = nn.Linear(self.num_test, self.num_test)
        self.layer_actor3 = nn.Linear(self.num_test, self.num_test)

        self.layer_merge = nn.Sequential()
        self.layer_merge.add_module('lm1',nn.Linear(self.num_test*2,self.num_units))
        self.layer_merge.add_module('lmf1',nn.LeakyReLU())
        self.layer_merge.add_module('lm2',nn.Linear(self.num_units,self.num_units))
        self.layer_merge.add_module('lmf2',nn.LeakyReLU())
        self.layer_merge.add_module('lm3',nn.Linear(self.num_units,self.num_outputs))

        self.critic_norm1 = nn.Sequential()
        self.critic_norm1.add_module('c_bn1', nn.LayerNorm(
                                            self.num_test))
        self.critic_norm1.add_module('c_bf1',nn.ReLU())

        self.critic_norm2 = nn.Sequential()
        self.critic_norm2.add_module('c_bn2', nn.LayerNorm(
                                            self.num_test))
        self.critic_norm2.add_module('c_bf2',nn.ReLU())


        self.actor_model = Actor_att1(args,with_action=True) 


    def forward(self,o,u,logger=None, iter_n=None):
        #n*[none,sdim], n*[none,adim]
        # print(o[0].shape,u[0].shape)
        state = torch.cat(o, dim=1) #[None,n_good*obs_size]
        action = torch.cat(u, dim=1) #[None,n_good*2]
        input = torch.cat([state, action], dim=1) #[None,n_good*(obs_size+2)]
        # print(input.shape)

        input_action = input[:, -2*(self.n_good):]
        self_action = input_action[:, self.index*2: (self.index+1)*2]
        good_action = input_action[:, :]

        # split self obs
        length_obs = 4+self.n_food*2+(self.n_good-1)*4
        self_start = (self.index)*length_obs

        # self mlp
        input_obs_self = input[:, self_start:self_start+length_obs]
        self_in = input_obs_self
        # self_in = torch.cat([self_in, self_action], 1) #[None,]

        # print(self_in.shape)
        self_out = self.actor_model.forward(self_in,self_action) #[None,self.num_test]
        # print(self_out.shape)

        # other agent mlp
        other_good_ins = []
        other_good_action = []
        for i in range(self.n_good):
            if i==self.index:
                continue
            other_good_beg = i*length_obs
            other_good_in = input[:,other_good_beg:other_good_beg+length_obs]
            # tmp = torch.cat([other_good_in, good_action[:, i*2:(i+1)*2]], 1)
            # print(tmp.shape)
            other_good_ins.append(other_good_in)
            other_good_action.append(good_action[:, i*2:(i+1)*2])

        batch_other_good_ins = torch.cat(other_good_ins, dim=0) #[n_other*None,size]
        batch_other_action = torch.cat(other_good_action,dim=0)
    
        # print(batch_other_good_ins[0,:,:].shape)
        other_good_outs = self.actor_model.forward(batch_other_good_ins,batch_other_action) #[n_other*None,self.num_test]
        other_good_outs = torch.split(other_good_outs,input.shape[0]) #n_other*[None,self.num_test]
        assert(len(other_good_outs)==(self.n_good-1))

        theta_out = []
        phi_out = []
        g_out = []

        theta_out.append(self.layer_actor1(self_out))
        phi_out.append(self.layer_actor2(self_out))
        g_out.append(self.layer_actor3(self_out))

        for i,out in enumerate(other_good_outs):
            theta_out.append(self.layer_actor1(out))
            phi_out.append(self.layer_actor2(out))
            g_out.append(self.layer_actor3(out))

        theta_outs = torch.stack(theta_out, 2) #[none, 16,3]
        assert(theta_outs.shape[2]==self.n_good)
        # print(theta_outs.shape)
        
        # print(theta_outs.shape)
        phi_outs = torch.stack(phi_out, 2)

        g_outs = torch.stack(g_out, 2)
        
        # print(torch.matmul(theta_outs, phi_outs.permute(0,2,1)).shape)
        # [None,self.num_test,n_good]x[None,n_good,num_test]
        self_attention = F.softmax(torch.matmul(theta_outs, phi_outs.permute(0,2,1))/math.sqrt(self.num_test),dim=2) # [None,self.num_test,num_test] x 
        # print(g_outs.shape,'self_attention')
        input_all = torch.matmul(self_attention, g_outs) #[None,self.num_test,n_good]
        input_all_new = []

        # print(input_all[:,:,0].shape)
        
        for i in range(self.n_good):
            # print(input_all[:,:,i].shape)
            input_all_new.append(self.critic_norm1(input_all[:,:,i])) #[None,self.numtest]
            # print(input_all[:,:,i].shape)
        
        input_all = torch.stack(input_all_new, 2) # [None,self.numtest,n_good]
        # print(input_all.shape)
        # input_all = tf.contrib.layers.layer_norm(input_all)

        self_out_new = input_all[:,:,0] #[None, num_test]
        good_out_new = input_all[:,:,1:self.n_good] #[None,num_test,n_other]

        #[None,1,num_test] x [None,num_test,n_other] -> [None,1,n_other]
        other_good_out_attn = F.softmax(torch.matmul(self_out_new.unsqueeze(1), good_out_new)/math.sqrt(self.num_test),dim=2) 
        
        if logger is not None:
            logger.add_scalars('agent%i/critic_attention' % self.index,
                                   dict(('attention_weight%i_' % h_i, other_good_out_attn[0,0,h_i]) for h_i 
                                        in range(self.n_good-1)),
                                   iter_n)

        #[None,1,n_other] x [None,n_other,num_test] -> [None,1,num_test]->[None,num_test]
        other_good_out = torch.matmul(other_good_out_attn, good_out_new.permute(0,2,1)).squeeze(1)
        # print(other_good_out.shape)
        other_good_out = self.critic_norm2(other_good_out)
        # print(other_good_out.shape)

        # merge layer for all
        input_merge = torch.cat([self_out, other_good_out], 1)
        # print(input_merge.shape)

        out = self.layer_merge(input_merge)

        return out


#MAAC+bare_actor1
class Attention_Critic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, args, hidden_dim=32, attend_heads=1, norm_in = True):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(Attention_Critic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.nagents = args.n_agents
        self.attend_heads = attend_heads
        self.args = args

        self.sa_encoder = Attention_Actor(args, hidden_dim, norm_in = True, with_action = True) 
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in zip(self.args.obs_shape,self.args.action_shape):
            idim = sdim + adim
            odim = adim
            
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()

        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.sa_encoder]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.args.n_agents)

    def forward(self, s, a, regularize=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        states = [s[i] for i in range(self.args.n_agents)] # [[None,sdim]]
        actions = [a[i] for i in range(self.args.n_agents)] # [[None,adim]]
        
        # extract state-action encoding for each agent 
        s2 = states # n* [None ,sdim]
        a2 = actions
        sa_encodings = self.sa_encoder(s2,a2) #[[None,hdim]] 
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in range(self.args.n_agents)] #[[None,hdim]]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors] #[[None,attdim]]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors] #[None,attdim]

        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings)]
                              for sel_ext in self.selector_extractors] #[[None,attdim]]

        other_all_values = [[] for _ in range(self.args.n_agents)]
        all_attend_logits = [[] for _ in range(self.args.n_agents)]
        all_attend_probs = [[] for _ in range(self.args.n_agents)]

        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, selector in zip(range(self.args.n_agents), curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != i]
                values = [v for j, v in enumerate(curr_head_values) if j != i]
                # calculate attention across agents
                #[None,1,attdim]x[None, attdim, n-1]-> [None,1,n-1]
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                #[None,attdim,n-1] x [None,1,n-1]-> [None,attdim]
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        # calculate Q per agent
        all_rets = []
        for i in range(self.args.n_agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[i](critic_in)
            int_acs = actions[i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)

            agent_rets.append(q)
        
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            
            if logger is not None:
                logger.add_scalars('agent%i/attention' % i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            all_rets.append(agent_rets)
       
        return all_rets


#PC_shared attention actor
class Critic_attf(nn.Module):
    def __init__(self, args):
        super(Critic_attf,self).__init__()

        self.args = args
        self.num_units = 32
        self.num_test = self.num_units // 2
        self.n_good = args.n_agents
        self.n_food = args.n_agents
        self.num_outputs = 1

        self.critics = nn.ModuleList()

        self.query_encoder = nn.Sequential()
        self.query_encoder.add_module("query_e", nn.Linear(self.num_test, self.num_test))

        self.key_encoder = nn.Sequential()
        self.key_encoder.add_module("key_e",nn.Linear(self.num_test, self.num_test))

        self.val_encoder = nn.Sequential()
        self.val_encoder.add_module('val_e',nn.Linear(self.num_test, self.num_test))
        
        self.sa_encoder = Actor_attf_single(args,with_action=True)

        self.critic_norm1 = nn.Sequential()
        self.critic_norm1.add_module('c_bn1', nn.LayerNorm(
                                            self.num_test))
        self.critic_norm1.add_module('c_bf1',nn.ReLU())

        self.critic_norm2 = nn.Sequential()
        self.critic_norm2.add_module('c_bn2', nn.LayerNorm(
                                            self.num_test))
        self.critic_norm2.add_module('c_bf2',nn.ReLU())


        for i in range(self.n_good):
            critic_layer = nn.Sequential()
            critic_layer.add_module('lm1',nn.Linear(self.num_test*2,self.num_units))
            critic_layer.add_module('lmf1',nn.LeakyReLU())
            critic_layer.add_module('lm2',nn.Linear(self.num_units,self.num_units))
            critic_layer.add_module('lmf2',nn.LeakyReLU())
            critic_layer.add_module('lm3',nn.Linear(self.num_units,self.num_outputs))
            self.critics.append(critic_layer)

        self.shared_modules = [self.query_encoder, self.key_encoder,
                               self.val_encoder, self.sa_encoder,self.critic_norm1,self.critic_norm2]


    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.n_good)

    def forward(self,o,u,logger=None, iter_n=None):
        #n*[none,sdim], n*[none,adim]
        # print(o[0].shape,u[0].shape)
        state = torch.cat(o, dim=1) #[None,n_good*obs_size]
        action = torch.cat(u, dim=1) #[None,n_good*2]
        input = torch.cat([state, action], dim=1) #[None,n_good*(obs_size+2)]
        # print(input.shape)
        out_list = []

        for index in range(self.n_good):
            input_action = input[:, -2*(self.n_good):]
            self_action = input_action[:, index*2: (index+1)*2]
            good_action = input_action[:, :]

            # split self obs
            length_obs = 4+self.n_food*2+(self.n_good-1)*4
            self_start = (index)*length_obs

            # self mlp
            input_obs_self = input[:, self_start:self_start+length_obs]
            self_in = input_obs_self
            # self_in = torch.cat([self_in, self_action], 1) #[None,]

            # print(self_in.shape)
            self_out = self.sa_encoder.forward(self_in,self_action) #[None,self.num_test]
            # print(self_out.shape)

            # other agent mlp
            other_good_ins = []
            other_good_action = []
            for i in range(self.n_good):
                if i==index:
                    continue
                other_good_beg = i*length_obs
                other_good_in = input[:,other_good_beg:other_good_beg+length_obs]
                # tmp = torch.cat([other_good_in, good_action[:, i*2:(i+1)*2]], 1)
                # print(tmp.shape)
                other_good_ins.append(other_good_in)
                other_good_action.append(good_action[:, i*2:(i+1)*2])

            batch_other_good_ins = torch.cat(other_good_ins, dim=0) #[n_other*None,size]
            batch_other_action = torch.cat(other_good_action,dim=0)
        
            # print(batch_other_good_ins[0,:,:].shape)
            other_good_outs = self.sa_encoder.forward(batch_other_good_ins,batch_other_action) #[n_other*None,self.num_test]
            other_good_outs = torch.split(other_good_outs,input.shape[0]) #n_other*[None,self.num_test]
            assert(len(other_good_outs)==(self.n_good-1))

            theta_out = []
            phi_out = []
            g_out = []

            theta_out.append(self.query_encoder(self_out))
            phi_out.append(self.key_encoder(self_out))
            g_out.append(self.val_encoder(self_out))

            for i,out in enumerate(other_good_outs):
                theta_out.append(self.query_encoder(out))
                phi_out.append(self.key_encoder(out))
                g_out.append(self.val_encoder(out))

            theta_outs = torch.stack(theta_out, 2) #[none, 16,3]
            assert(theta_outs.shape[2]==self.n_good)
            # print(theta_outs.shape)
            
            # print(theta_outs.shape)
            phi_outs = torch.stack(phi_out, 2)

            g_outs = torch.stack(g_out, 2)
            
            # print(torch.matmul(theta_outs, phi_outs.permute(0,2,1)).shape)
            # [None,self.num_test,n_good]x[None,n_good,num_test]
            self_attention = F.softmax(torch.matmul(theta_outs, phi_outs.permute(0,2,1))/math.sqrt(self.num_test),dim=2) # [None,self.num_test,num_test] x 
            # print(g_outs.shape,'self_attention')
            input_all = torch.matmul(self_attention, g_outs) #[None,self.num_test,n_good]
            input_all_new = []

            # print(input_all[:,:,0].shape)
            
            for i in range(self.n_good):
                # print(input_all[:,:,i].shape)
                input_all_new.append(self.critic_norm1(input_all[:,:,i])) #[None,self.numtest]
                # print(input_all[:,:,i].shape)
            
            input_all = torch.stack(input_all_new, 2) # [None,self.numtest,n_good]
            # print(input_all.shape)
            # input_all = tf.contrib.layers.layer_norm(input_all)

            self_out_new = input_all[:,:,0] #[None, num_test]
            good_out_new = input_all[:,:,1:self.n_good] #[None,num_test,n_other]

            #[None,1,num_test] x [None,num_test,n_other] -> [None,1,n_other]
            other_good_out_attn = F.softmax(torch.matmul(self_out_new.unsqueeze(1), good_out_new)/math.sqrt(self.num_test),dim=2) 
            
            if logger is not None:
                logger.add_scalars('agent%i/critic_attention' % index,
                                    dict(('attention_weight%i_' % h_i, other_good_out_attn[0,0,h_i]) for h_i 
                                            in range(self.n_good-1)),
                                    iter_n)

            #[None,1,n_other] x [None,n_other,num_test] -> [None,1,num_test]->[None,num_test]
            other_good_out = torch.matmul(other_good_out_attn, good_out_new.permute(0,2,1)).squeeze(1)
            # print(other_good_out.shape)
            other_good_out = self.critic_norm2(other_good_out)
            # print(other_good_out.shape)

            # merge layer for all
            input_merge = torch.cat([self_out, other_good_out], 1)
            # print(input_merge.shape)

            out = self.critics[index](input_merge)
            out_list.append(out)

        return out_list

class Actor_attf_single(nn.Module):
    def __init__(self, args, with_action = False):
        super(Actor_attf_single,self).__init__()
        ######################################
        self.num_test = 32 // 2
        self.num_units = 32
        self.n_good = args.n_agents
        self.n_food = args.n_agents
        self.num_outputs = self.num_test
        self.with_action = with_action
        self.self_size = 4
        self.other_size = 4
        self.food_size = 2

        self.en_encoder = nn.Sequential()
        if self.with_action:
            self.en_encoder.add_module('en_coder1',nn.Linear(self.self_size+2, self.num_units))
        else:
            self.en_encoder.add_module('en_coder1',nn.Linear(self.self_size, self.num_units))

        self.en_encoder.add_module('en_af1',nn.ReLU())
        self.en_encoder.add_module('en_coder2',nn.Linear(self.num_units, self.num_test))
        self.en_encoder.add_module('en_af2',nn.ReLU())

        self.oa_encoder = nn.Sequential()
        self.oa_encoder.add_module('oa_coder1',nn.Linear(self.other_size, self.num_units))
        self.oa_encoder.add_module('oa_af1',nn.ReLU())
        self.oa_encoder.add_module('oa_coder2',nn.Linear(self.num_units, self.num_test))
        self.oa_encoder.add_module('oa_af2',nn.ReLU())

        self.oa_attention = nn.Sequential()
        self.oa_attention.add_module('oa_bn', nn.LayerNorm(
                                            self.num_test))
        self.oa_attention.add_module('oa_bf1',nn.ReLU())

        self.goal_encoder = nn.Sequential()
        self.goal_encoder.add_module('goal_coder1',nn.Linear(self.food_size, self.num_units))
        self.goal_encoder.add_module('goal_af1',nn.ReLU())
        self.goal_encoder.add_module('goal_coder2',nn.Linear(self.num_units, self.num_test))
        self.goal_encoder.add_module('goal_af2',nn.ReLU())

        self.goal_attention = nn.Sequential()
        self.goal_attention.add_module('goal_bn', nn.LayerNorm(
                                            self.num_test))
        self.goal_attention.add_module('goal_bf1',nn.ReLU())

        self.actor_layer = nn.Sequential()
        if self.with_action:
            self.actor_layer.add_module('merge_l1',nn.Linear(self.num_test*3,self.num_units))
            self.actor_layer.add_module('merge_f1',nn.ReLU())
            self.actor_layer.add_module('merge_l2',nn.Linear(self.num_units,self.num_units))
            self.actor_layer.add_module('merge_f2',nn.ReLU())
            self.actor_layer.add_module('merge_l3',nn.Linear(self.num_units,self.num_outputs))
        else:
            self.actor_layer.add_module('merge_l1',nn.Linear(self.num_test*3,self.num_units))
            self.actor_layer.add_module('merge_f1', nn.LeakyReLU())
            self.actor_layer.add_module('merge_l2',nn.Linear(self.num_units,self.num_units))
            self.actor_layer.add_module('merge_f2', nn.LeakyReLU())
            self.actor_layer.add_module('merge_l3', nn.Linear(self.num_units,2))
            self.actor_layer.add_module('merge_f3', nn.Tanh())

    def forward(self, s_input, a_input = None, logger=None, agent_id=None, iter_n=None):
        #self mlp
        # print(s_input.shape) #[None, sdim]
        
        self_in = s_input[:,:4] 
        
        if self.with_action:
            #obs_self+ food+other_obs+self_action
            self_action = a_input[:,:]
            # print(self_action)
            self_in = torch.cat([self_in, self_action], dim=1) #[None, 6]
            # print(self_in.shape) 

        # print("self",self_in.shape)
        self_out = self.en_encoder(self_in) #[None,self.num_test]

        #other mlp
        other_good_in = s_input[:, 4:4+(self.n_good-1)*4]
        other_good_ins = []
        for i in range(self.n_good-1):
            temp1 = other_good_in[:, i*2:2+i*2] #[None,2] v  v+p 
            temp2 = other_good_in[:, (self.n_good-1)*2+i*2:(self.n_good-1)*2+i*2+2] #p
            temp = torch.cat([temp1,temp2],dim=1) #[none,4]
            assert(temp.shape[1]==4)
            other_good_ins.append(temp)

        other_good_ins = torch.stack(other_good_ins, dim=0) #[n_good-1,None,4]
        # print(other_good_ins.shape)
        
        other_outs = self.oa_encoder(other_good_ins) #[n_good-1,None,self.num_test]
        other_good_out = other_outs.permute(1, 2, 0) #[None,self.num_test,n_good-1]

        other_good_out_attn = F.softmax(torch.matmul(self_out.unsqueeze(1), other_good_out)/math.sqrt(self.num_test),dim=2) # [None,1,self.num_test]x[None,self.num_test,n_good-1]-> [None,1,n_other]

        if logger is not None:
            # print(other_good_out_attn.shape)
            logger.add_scalars('agent%i/actor_attention/other_agents' % agent_id,
                                   dict(('attention_weight%i_' % h_i, other_good_out_attn[0,0,h_i]) for h_i 
                                        in range(self.n_good-1)),
                                   iter_n)
        # print("attn:", other_good_out_attn.shape)
        # [None,1,n_other]x[None,n_other,self.num_test]->[None, 1,self.num_test]->[None,self.num_test]
        other_good_out = torch.matmul(other_good_out_attn, other_good_out.permute(0,2,1)).squeeze(1)
        
        other_good_out = self.oa_attention(other_good_out) #[None,self.num_test]
        # print(other_good_out.shape)

        #food mlp
        food_input = s_input[:,-self.n_food*2:] #[None,n_food*2]

        food_input = torch.split(food_input, 2, dim=1) #split size = 2 tuple()
        # print(len(food_input))
        food_input = torch.stack(list(food_input), dim=0) #[n_food,None,2]
        
        food_outs = self.goal_encoder(food_input) #[n_food,None, self.num_test]

        food_out = food_outs.permute(1, 2, 0) #[None,self.num_test,n_food]
        # print(torch.matmul(self_out.unsqueeze(1), food_out).shape)
        food_out_attn = F.softmax(torch.matmul(self_out.unsqueeze(1), food_out)/math.sqrt(self.num_test),dim=2) #[None, 1, self.num_test] x [None,self.num_test,n_food]->[None,1,n_food]

        if logger is not None:
            logger.add_scalars('agent%i/actor_attention/goals' % agent_id,
                                   dict(('attention_weight%i_' % h_i, food_out_attn[0,0,h_i]) for h_i 
                                        in range(self.n_good)),
                                   iter_n)

        food_out = torch.matmul(food_out_attn, food_out.permute(0,2,1)).squeeze(1)  #[None,self.num_test]
        food_out = self.goal_attention(food_out) 
        # print(food_out.shape)

        input_merge = torch.cat([self_out, food_out, other_good_out], 1) #[None,self.num_test*3]
        assert(input_merge.shape[1]==self.num_test*3)
        # print(input_merge.shape)
        
        out = self.actor_layer(input_merge)
        
        return out


class Actor_attf(nn.Module):
    def __init__(self, args, with_action = False):
        super(Actor_attf,self).__init__()

        ######################################
        self.num_test = 32 // 2
        self.num_units = 32
        self.n_good = args.n_agents
        self.n_food = args.n_agents
        self.num_outputs = self.num_test
        self.with_action = with_action
        self.self_size = 4
        self.other_size = 4
        self.food_size = 2

        self.en_encoder = nn.Sequential()
        if self.with_action:
            self.en_encoder.add_module('en_coder1',nn.Linear(self.self_size+2, self.num_units))
        else:
            self.en_encoder.add_module('en_coder1',nn.Linear(self.self_size, self.num_units))

        self.en_encoder.add_module('en_af1',nn.ReLU())
        self.en_encoder.add_module('en_coder2',nn.Linear(self.num_units, self.num_test))
        self.en_encoder.add_module('en_af2',nn.ReLU())

        self.oa_encoder = nn.Sequential()
        self.oa_encoder.add_module('oa_coder1',nn.Linear(self.other_size, self.num_units))
        self.oa_encoder.add_module('oa_af1',nn.ReLU())
        self.oa_encoder.add_module('oa_coder2',nn.Linear(self.num_units, self.num_test))
        self.oa_encoder.add_module('oa_af2',nn.ReLU())

        self.oa_attention = nn.Sequential()
        self.oa_attention.add_module('oa_bn', nn.LayerNorm(
                                            self.num_test))
        self.oa_attention.add_module('oa_bf1',nn.ReLU())

        self.goal_encoder = nn.Sequential()
        self.goal_encoder.add_module('goal_coder1',nn.Linear(self.food_size, self.num_units))
        self.goal_encoder.add_module('goal_af1',nn.ReLU())
        self.goal_encoder.add_module('goal_coder2',nn.Linear(self.num_units, self.num_test))
        self.goal_encoder.add_module('goal_af2',nn.ReLU())

        self.goal_attention = nn.Sequential()
        self.goal_attention.add_module('goal_bn', nn.LayerNorm(
                                            self.num_test))
        self.goal_attention.add_module('goal_bf1',nn.ReLU())

        self.actor_layers = nn.ModuleList()
        for i in range(self.n_good):
            actor_layer = nn.Sequential()
            if self.with_action:
                actor_layer.add_module('merge_l1',nn.Linear(self.num_test*3,self.num_units))
                actor_layer.add_module('merge_f1',nn.ReLU())
                actor_layer.add_module('merge_l2',nn.Linear(self.num_units,self.num_units))
                actor_layer.add_module('merge_f2',nn.ReLU())
                actor_layer.add_module('merge_l3',nn.Linear(self.num_units,self.num_outputs))
            else:
                actor_layer.add_module('merge_l1',nn.Linear(self.num_test*3,self.num_units))
                actor_layer.add_module('merge_f1', nn.LeakyReLU())
                actor_layer.add_module('merge_l2',nn.Linear(self.num_units,self.num_units))
                actor_layer.add_module('merge_f2', nn.LeakyReLU())
                actor_layer.add_module('merge_l3', nn.Linear(self.num_units,2))
                actor_layer.add_module('merge_f3', nn.Tanh())
            self.actor_layers.append(actor_layer)

        self.shared_modules = [self.en_encoder, self.goal_encoder,
                               self.oa_encoder, self.goal_attention,self.oa_attention]

        #####################################

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.n_good)    

    def forward(self, s_input, a_input = None, logger=None, iter_n=None):
        #self mlp
        # print(s_input.shape) #[n, None, sdim]
        # s_input = torch.tensor(s_input)
        # a_input = torch.tensor(a_input)
        if isinstance(s_input,list):
            s_input = torch.stack(s_input,dim=0)
        
        out_list = []
        for i in range(self.n_good):
            self_in = s_input[i,:,:4] 
            
            if self.with_action:
                #obs_self+ food+other_obs+self_action
                self_action = a_input[i, :,:]
                # print(self_action)
                self_in = torch.cat([self_in, self_action], dim=1) #[None, 6]
                # print(self_in.shape) 

            # print("self",self_in.shape)
            self_out = self.en_encoder(self_in) #[None,self.num_test]

            #other mlp
            other_good_in = s_input[i,:, 4:4+(self.n_good-1)*4]
            other_good_ins = []
            for i in range(self.n_good-1):
                temp1 = other_good_in[:, i*2:2+i*2] #[None,2] v  v+p 
                temp2 = other_good_in[:, (self.n_good-1)*2+i*2:(self.n_good-1)*2+i*2+2] #p
                temp = torch.cat([temp1,temp2],dim=1) #[none,4]
                assert(temp.shape[1]==4)
                other_good_ins.append(temp)

            other_good_ins = torch.stack(other_good_ins, dim=0) #[n_good-1,None,4]
            # print(other_good_ins.shape)
            
            other_outs = self.oa_encoder(other_good_ins) #[n_good-1,None,self.num_test]
            other_good_out = other_outs.permute(1, 2, 0) #[None,self.num_test,n_good-1]

            other_good_out_attn = F.softmax(torch.matmul(self_out.unsqueeze(1), other_good_out)/math.sqrt(self.num_test),dim=2) # [None,1,self.num_test]x[None,self.num_test,n_good-1]-> [None,1,n_other]

            if logger is not None:
                # print(other_good_out_attn.shape)
                logger.add_scalars('agent%i/actor_attention/other_agents'%i,
                                        dict(('attention_weight%i_' % h_i, other_good_out_attn[0,0,h_i]) for h_i 
                                            in range(self.n_good-1)),
                                        iter_n)
            # print("attn:", other_good_out_attn.shape)
            # [None,1,n_other]x[None,n_other,self.num_test]->[None, 1,self.num_test]->[None,self.num_test]
            other_good_out = torch.matmul(other_good_out_attn, other_good_out.permute(0,2,1)).squeeze(1)
            
            other_good_out = self.oa_attention(other_good_out) #[None,self.num_test]
            # print(other_good_out.shape)

            #food mlp
            food_input = s_input[i,:,-self.n_food*2:] #[None,n_food*2]

            food_input = torch.split(food_input, 2, dim=1) #split size = 2 tuple()
            # print(len(food_input))
            food_input = torch.stack(list(food_input), dim=0) #[n_food,None,2]
            
            food_outs = self.goal_encoder(food_input) #[n_food,None, self.num_test]

            food_out = food_outs.permute(1, 2, 0) #[None,self.num_test,n_food]
            # print(torch.matmul(self_out.unsqueeze(1), food_out).shape)
            food_out_attn = F.softmax(torch.matmul(self_out.unsqueeze(1), food_out)/math.sqrt(self.num_test),dim=2) #[None, 1, self.num_test] x [None,self.num_test,n_food]->[None,1,n_food]

            if logger is not None:
                logger.add_scalars('agent%i/actor_attention/goals'%i,
                                        dict(('attention_weight%i_' % h_i, food_out_attn[0,0,h_i]) for h_i 
                                            in range(self.n_good)),
                                        iter_n)

            food_out = torch.matmul(food_out_attn, food_out.permute(0,2,1)).squeeze(1)  #[None,self.num_test]
            food_out = self.goal_attention(food_out) 
            # print(food_out.shape)

            input_merge = torch.cat([self_out, food_out, other_good_out], 1) #[None,self.num_test*3]
            assert(input_merge.shape[1]==self.num_test*3)
            # print(input_merge.shape)
        
            out1 = self.actor_layers[i](input_merge)
            out_list.append(out1)
        
        return out_list