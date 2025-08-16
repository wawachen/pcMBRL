from typing import DefaultDict
import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class Actor_NN(nn.Module):
    def __init__(self, args, with_action=False):
        super(Actor_NN, self).__init__()
        # self.total_size = args.obs_shape[agent_id]
        self.hidden_size = [64]
        self.feature_size = [64]
        self.n = 2*args.n_agents-1
        self.with_action = with_action
        self.args = args

        self.self_size = 2
        self.mean_embeding_dim_food = 2
        self.mean_embeding_dim_other = 2
        self.mean_embeding_size = (args.n_agents-1)*self.mean_embeding_dim_other+args.n_agents*self.mean_embeding_dim_food

        self.fc1 = nn.Linear(self.mean_embeding_dim_food, self.feature_size[0]) #food
        self.fc2 = nn.Linear(self.mean_embeding_dim_other, self.feature_size[0]) #other
        if with_action:
            self.fc3 = nn.Linear(self.self_size+self.feature_size[0]+2, self.hidden_size[0]) #merge
            self.value_fc = nn.Linear(self.hidden_size[0], 1) 
        else:
            self.fc3 = nn.Linear(self.self_size+self.feature_size[0], self.hidden_size[0]) #merge
            self.action_out1 = nn.Linear(self.hidden_size[0], 1) #angular velocity [-1,1]
            self.action_out2 = nn.Linear(self.hidden_size[0], 1) #linear velocity [0,1]


    def forward(self, x):
        self_input = x[:, :self.self_size]
        food_input = x[:, self.self_size:self.self_size+self.args.n_agents*self.mean_embeding_dim_food].view(-1,self.args.n_agents,self.mean_embeding_dim_food)
        other_input = x[:,self.mean_embeding_size-self.mean_embeding_dim_other*(self.args.n_agents-1):self.mean_embeding_size].view(-1,self.args.n_agents-1,self.mean_embeding_dim_other)

        if self.with_action:
            action_input = x[:,-2:]

        self_out = F.relu(self.fc1(food_input))
        other_out = F.relu(self.fc2(other_input))
        combo_out = torch.cat([self_out,other_out],dim=1)
        combo_out_sum = torch.sum(combo_out,1)
        combo_out_last = torch.div(combo_out_sum,self.n)

        if self.with_action:
            last_out = torch.cat([combo_out_last, self_input,action_input], dim=1)
            last_out = F.relu(self.fc3(last_out))

            value = self.value_fc(last_out)
            return value
        else:
            # print(combo_out_last.shape,self_input.shape)
            last_out = torch.cat([combo_out_last, self_input], dim=1)
            last_out = F.relu(self.fc3(last_out))

            act1 = torch.tanh(self.action_out1(last_out))
            act2 = torch.sigmoid(self.action_out2(last_out))

            actions = torch.cat((act1,act2),dim=-1)
            # print(actions)
            return actions


class Critic_NN(nn.Module):
    def __init__(self, args):
        super(Critic_NN, self).__init__()
        self.n_food = args.n_agents
        self.n_good = args.n_agents

        self.self_size = 2
        self.mean_embeding_dim_food = 2
        self.mean_embeding_dim_other = 2
        self.mean_embeding_size = (args.n_agents-1)*self.mean_embeding_dim_other+args.n_agents*self.mean_embeding_dim_food

        self.actor_NN = Actor_NN(args,with_action=True)

    def forward(self, state, action):
        state = torch.cat(state, dim=1) #[None, n*s_dim]
        action = torch.cat(action, dim=1) #[None,n*act_dim]
        input = torch.cat([state, action], dim=1) #[None,n*(s_dim+act_dim)]

        input_action = input[:, -2*(self.n_good):] 
        good_action = input_action[:, :]

        # split self obs
        length_obs = self.self_size+self.mean_embeding_size

        good_ins = []
        for i in range(self.n_good):
            good_beg = i*length_obs
            good_in = input[:,good_beg:good_beg+length_obs]
            tmp = torch.cat([good_in, good_action[:, i*2:(i+1)*2]], 1) #[None,actor_input]
            # print(tmp.shape)
            good_ins.append(tmp)
        good_out = torch.cat(good_ins,dim=0) #[n,None,actor_input]
        print(good_out.shape)
        batch_out = self.actor_NN.forward(good_out) #[n*None,1]
        batch_out = batch_out.view(self.n_good,-1,1) #[n,None,1]
        print(batch_out.shape)
        batch_out = batch_out.permute(1,0,2)
        bacth_sum = torch.sum(batch_out,1) #[None,1]
        bacth_last = torch.div(bacth_sum,self.n_good)
        q_value = bacth_last
        print("q_value",q_value)
        return q_value
