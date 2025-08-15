from typing import DefaultDict
import torch 
import math 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


#--------------drone--------------------------------
#PC_shared attention actor
class Critic_attf(nn.Module):
    def __init__(self, args):
        super(Critic_attf,self).__init__()

        self.num_units = 32
        self.num_test = self.num_units // 2
        self.n_good = args.n_agents
        self.n_food = args.n_agents
        self.num_outputs = 1

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

        self.critic_layer = nn.Sequential()
        self.critic_layer.add_module('lm1',nn.Linear(self.num_test*2,self.num_units))
        self.critic_layer.add_module('lmf1',nn.LeakyReLU())
        self.critic_layer.add_module('lm2',nn.Linear(self.num_units,self.num_units))
        self.critic_layer.add_module('lmf2',nn.LeakyReLU())
        self.critic_layer.add_module('lm3',nn.Linear(self.num_units,self.num_outputs))

        self.shared_modules = [self.query_encoder, self.key_encoder,
                               self.val_encoder, self.sa_encoder,self.critic_norm1,self.critic_norm2,self.critic_layer]


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

            out = self.critic_layer(input_merge)
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

        self.shared_modules = [self.en_encoder, self.goal_encoder,
                               self.oa_encoder, self.goal_attention,self.oa_attention,self.actor_layer]

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
            for j in range(self.n_good-1):
                temp1 = other_good_in[:, j*2:2+j*2] #[None,2] v  v+p 
                temp2 = other_good_in[:, (self.n_good-1)*2+j*2:(self.n_good-1)*2+j*2+2] #p
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
                logger.add_scalars('agent%d/actor_attention/other_agents'%i,
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
                logger.add_scalars('agent%d/actor_attention/goals'%i,
                                        dict(('attention_weight%i_' % h_i, food_out_attn[0,0,h_i]) for h_i 
                                            in range(self.n_good)),
                                        iter_n)

            food_out = torch.matmul(food_out_attn, food_out.permute(0,2,1)).squeeze(1)  #[None,self.num_test]
            food_out = self.goal_attention(food_out) 
            # print(food_out.shape)

            input_merge = torch.cat([self_out, food_out, other_good_out], 1) #[None,self.num_test*3]
            assert(input_merge.shape[1]==self.num_test*3)
            # print(input_merge.shape)
        
            out1 = self.actor_layer(input_merge)
            out_list.append(out1)
        
        return out_list