from typing import DefaultDict
import torch 
import math 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


#MAAC+bare_actor1
class Attention_Critic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, args, id, hidden_dim=32, attend_heads=1, norm_in = True):
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
        self.id = id

        #args, num_agents, num_entities, input_size=16, hidden_dim=128, embed_dim=None,pos_index=2, norm_in=False, nonlin=nn.ReLU, n_heads=1, mask_dist=None, with_action = False
        self.sa_encoder = Attention_Actor(args, input_size=4, with_action = True) 
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        
        sdim = self.args.obs_shape[0]
        adim = self.args.action_shape[0]
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
        s2 = torch.tensor(states) # n* [None ,sdim]
        a2 = torch.tensor(actions)
        s2 = s2.view(-1,s2.shape[2])
        a2 = a2.view(-1,a2.shape[2])
        sa_encodings = [self.sa_encoder(s2,a2,agent_id = self.id) for _ in range(self.args.n_agents)] #[[None,hdim]] 
        # extract state encoding for each agent that we're returning Q for
        s_encoding = self.state_encoders[0](states[self.id]) #[[None,hdim]]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors] #[[None,attdim]]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors] #[None,attdim]

        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(s_encoding)]
                              for sel_ext in self.selector_extractors] #[[None,attdim]]

        other_all_values = [[]]
        all_attend_logits = [[]]
        all_attend_probs = [[]]

        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):

            # iterate over agents
            i = self.id
            selector = curr_head_selectors[0]
           
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
            other_all_values[0].append(other_values)
            all_attend_logits[0].append(attend_logits)
            all_attend_probs[0].append(attend_weights)

        # calculate Q per agent
        critic_in = torch.cat((s_encoding, *other_all_values[0]), dim=1)
        all_q = self.critics[0](critic_in)
        int_acs = actions[self.id].max(dim=1, keepdim=True)[1]
        q = all_q.gather(1, int_acs)

        # if regularize:
        #     # regularize magnitude of attention logits
        #     attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
        #                                 all_attend_logits[i])
        #     q = attend_mag_reg
            
        
        if logger is not None:
            logger.add_scalars('agent%i/attention' % self.id,
                                dict(('head0_attention%i' % h_i, all_attend_probs[0][0][0][0][h_i]) for h_i 
                                    in range(self.id-1)),
                                niter)
        
       
        return q


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Attention_Actor(nn.Module):
    def __init__(self, args, input_size=16, hidden_dim=128, embed_dim=None,
                 norm_in=False, nonlin=nn.ReLU, n_heads=1, mask_dist=None, with_action = False):
        super().__init__()

        self.h_dim = hidden_dim
        self.nonlin = nonlin
        self.num_agents = args.n_agents # number of agents
        self.num_entities = args.n_agents # number of entities
        self.K = 3 # message passing rounds
        self.embed_dim = self.h_dim if embed_dim is None else embed_dim
        self.n_heads = n_heads
        self.mask_dist = mask_dist
        self.input_size = input_size
        self.with_action = with_action
        
        self.encoder = nn.Sequential(nn.Linear(self.input_size,self.h_dim),
                                     self.nonlin(inplace=True))

        self.messages = MultiHeadAttention(n_heads=self.n_heads,input_dim=self.h_dim,embed_dim=self.embed_dim)

        self.update = nn.Sequential(nn.Linear(self.h_dim+self.embed_dim,self.h_dim),
                                    self.nonlin(inplace=True))

        self.value_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                        self.nonlin(inplace=True),
                                        nn.Linear(self.h_dim,1))

        self.policy_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                         self.nonlin(inplace=True))

        
        self.entity_encoder = nn.Sequential(nn.Linear(2,self.h_dim),
                                            self.nonlin(inplace=True))
        
        self.entity_messages = MultiHeadAttention(n_heads=1,input_dim=self.h_dim,embed_dim=self.embed_dim)
        
        self.entity_update = nn.Sequential(nn.Linear(self.h_dim+self.embed_dim,self.h_dim),
                                            self.nonlin(inplace=True))
        
        # num_actions = action_space.n
        # self.dist = Categorical(self.h_dim,num_actions)

        self.is_recurrent = False

        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.input_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.apply(weights_init)

        self.attn_mat = np.ones((num_agents, num_agents))

        self.dropout_mask = None

        self.merge_layer = nn.Sequential()
        if not self.with_action:
            self.merge_layer.add_module('merge_l2',nn.Linear(self.h_dim,self.h_dim))
            self.merge_layer.add_module('merge_f2', nn.LeakyReLU())
            self.merge_layer.add_module('merge_l3', nn.Linear(self.h_dim,2))
            self.merge_layer.add_module('merge_f3', nn.Tanh())

    def calculate_mask(self, inp):
        # inp is batch_size x self.input_size where batch_size is num_processes*num_agents
        
        # pos = inp[:, self.pos_index:self.pos_index+2]
        bsz = inp.size(0)//self.num_agents
        mask = torch.full(size=(bsz,self.num_agents,self.num_agents),fill_value=0,dtype=torch.uint8)
        
        # if self.mask_dist is not None and self.mask_dist > 0: 
        #     for i in range(1,self.num_agents):
        #         shifted = torch.roll(pos,-bsz*i,0)
        #         dists = torch.norm(pos-shifted,dim=1)
        #         restrict = dists > self.mask_dist
        #         for x in range(self.num_agents):
        #             mask[:,x,(x+i)%self.num_agents].copy_(restrict[bsz*x:bsz*(x+1)])
        
        # elif self.mask_dist is not None and self.mask_dist == -10:
        #    if self.dropout_mask is None or bsz!=self.dropout_mask.shape[0] or np.random.random_sample() < 0.1: # sample new dropout mask
        #        temp = torch.rand(mask.size()) > 0.85
        #        temp.diagonal(dim1=1,dim2=2).fill_(0)
        #        self.dropout_mask = (temp+temp.transpose(1,2))!=0
        #    mask.copy_(self.dropout_mask)

        return mask            


    def forward(self, inp, a_input = None, logger=None, agent_id=None, iter_n=None):
        # inp should be (batch_size,input_size)
        # inp - {iden, vel(2), pos(2), entities(...)}
        agent_inp = inp[:,:self.input_size]          
        mask = self.calculate_mask(agent_inp) # shape <batch_size/N,N,N> with 0 for comm allowed, 1 for restricted

        if self.with_action:
            self_action = a_input[:,:]
            # print(self_action)
            agent_inp = torch.cat([agent_inp, self_action], dim=1) #[None, 6]
            assert(agent_inp.shape[1]==self.input_size+2)

        h = self.encoder(agent_inp) # should be (batch_size,self.h_dim)
       
        landmark_inp = inp[:,-self.num_entities*2:] # x,y pos of landmarks wrt agents
        # should be (batch_size,self.num_entities,self.h_dim)
        he = self.entity_encoder(landmark_inp.contiguous().view(-1,2)).view(-1,self.num_entities,self.h_dim) 
        entity_message = self.entity_messages(h.unsqueeze(1),he).squeeze(1) # should be (batch_size,self.h_dim)
        h = self.entity_update(torch.cat((h,entity_message),1)) # should be (batch_size,self.h_dim)

        h = h.view(self.num_agents,-1,self.h_dim).transpose(0,1) # should be (batch_size/N,N,self.h_dim)
        
        for k in range(self.K):
            m, attn = self.messages(h, mask=mask, return_attn=True) # should be <batch_size/N,N,self.embed_dim>
            h = self.update(torch.cat((h,m),2)) # should be <batch_size/N,N,self.h_dim>
        h = h.transpose(0,1).contiguous().view(-1,self.h_dim)

        if self.with_action:
            out = h
        else:
            out = self.merge_layer(h)
        
        self.attn_mat = attn.squeeze().detach().cpu().numpy()

        if logger is not None:
            # print(other_good_out_attn.shape)
            logger.add_scalars('agent%i/actor_attention/other_agents' % agent_id,
                                   dict(('attention_weight%i_' % h_i, self.attn_mat[agent_id, h_i]) for h_i 
                                        in range(self.num_agents)),
                                   iter_n)
        # print(self.attn_mat)
        return out # should be <batch_size, self.h_dim> again


class MultiHeadAttention(nn.Module):
    # taken from https://github.com/wouterkool/attention-tsp/blob/master/graph_encoder.py
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None, return_attn=False):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -math.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        
        if return_attn:
            return out, attn
        return out




