import torch
import os
from pyrep.networks.actor_critic import Actor1, Critic1
import numpy as np

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

class MADDPG:
    def __init__(self, args):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.train_step = 0

        # create the network
        self.actor_networks = [Actor1(args,i) for i in range(self.args.n_agents)]
        self.critic_network = Critic1(args)

        # build up the target network
        self.actor_target_networks = [Actor1(args,i) for i in range(self.args.n_agents)]
        self.critic_target_network = Critic1(args)

        # load the weights into the target networks
        for i,actor_target_network in enumerate(self.actor_target_networks):
            actor_target_network.load_state_dict(self.actor_networks[i].state_dict())

        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optims = [torch.optim.Adam(self.actor_networks[i].parameters(), lr=self.args.lr_actor) for i in range(self.args.n_agents)]
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/params.pkl'):
            self.initialise_networks(self.model_path+'/params.pkl')
            print('successfully loaded actor_critic_network')
            
    # soft update
    def _soft_update_target_network(self):
        for i in range(self.args.n_agents):
            for target_param, param in zip(self.actor_target_networks[i].parameters(), self.actor_networks[i].parameters()):
                target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions,w_id):
        #[none*(s,a,r,s1)]
        o, u, o_next = [[] for _ in range(self.args.n_agents)], [[] for _ in range(self.args.n_agents)], [[] for _ in range(self.args.n_agents)]  # 用来装每个agent经验中的各项
        r = []

        for transition in transitions:
            s_c = transition[0]
            u_c = transition[1]
            r_c = transition[2]
            s_nc = transition[3]
            # print("obs",s_c, "u",u_c)
            for agent_id in range(self.args.n_agents):
                # print(agent_id,s_c.shape,s_c)
                o[agent_id].append(s_c[agent_id,:])
                u[agent_id].append(u_c[agent_id,:])
                o_next[agent_id].append(s_nc[agent_id,:])
            r.append(r_c[0])

        for i in range(self.args.n_agents):
            o[i] = torch.tensor(o[i], dtype=torch.float32)
            u[i] = torch.tensor(u[i], dtype=torch.float32)
            o_next[i] = torch.tensor(o_next[i], dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            for agent_id in range(self.args.n_agents):
                u_next.append(self.actor_target_networks[agent_id](o_next[agent_id]))
                
            q_next = self.critic_target_network(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        # print(w_id)
        critic_loss = ((target_q - q_value).pow(2)*torch.tensor(w_id)).mean()
        delta = target_q - q_value

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        actor_loss = []
        for agent_id in range(self.args.n_agents): 
            uid = u.copy()
            uid[agent_id] = self.actor_networks[agent_id](o[agent_id])
            actor_loss.append(- self.critic_network(o, uid).mean()) 
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # # update the network
        for agent_id in range(self.args.n_agents):
            self.actor_optims[agent_id].zero_grad()
            actor_loss[agent_id].backward()
            self.actor_optims[agent_id].step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

        return delta,self.train_step

   
    def initialise_networks(self, path):
        
        checkpoint = torch.load(path) # load the torch data

        for agent_id in range(self.args.n_agents):
            self.actor_networks[agent_id].load_state_dict(checkpoint['actor_params'][agent_id])    # actor parameters
            self.actor_target_networks[agent_id].load_state_dict(checkpoint['actor_target_params'][agent_id]) 
            self.actor_optims[agent_id].load_state_dict(checkpoint['actor_optim_params'][agent_id]) # actor optimiser state

        self.critic_network.load_state_dict(checkpoint['critic_params'])    # critic parameters
        self.critic_optim.load_state_dict(checkpoint['critic_optim_params']) # critic optimiser state
        self.critic_target_network.load_state_dict(checkpoint['critic_target_params']) 
         

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        save_dict = {'actor_params' : [actor_network.state_dict() for actor_network in self.actor_networks],
                    'actor_optim_params': [actor_optim.state_dict() for actor_optim in self.actor_optims],
                    'actor_target_params': [actor_target_network.state_dict() for actor_target_network in self.actor_target_networks],
                    'critic_params' : self.critic_network.state_dict(),
                    'critic_optim_params' : self.critic_optim.state_dict(),
                    'critic_target_params' : self.critic_target_network.state_dict()}

        torch.save(save_dict, model_path + '/' + num + '_params.pkl')




