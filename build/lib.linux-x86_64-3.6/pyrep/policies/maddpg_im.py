import torch
import os
from pyrep.networks.actor_critic import Actor_im, Critic_im

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor_im(args).to(device)
        self.critic_network = Critic_im(args).to(device)

        # build up the target network
        self.actor_target_network = Actor_im(args).to(device)
        self.critic_target_network = Critic_im(args).to(device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor_im)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic_im)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        # print((self.model_path + '/actor_params.pkl'))
        # if os.path.exists(self.model_path + '/actor_params.pkl'):
        #     self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
        #     self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
        #     print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
        #                                                                   self.model_path + '/actor_params.pkl'))
        #     print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
        #                                                                    self.model_path + '/critic_params.pkl'))

        if os.path.exists(self.model_path + '/params.pkl'):
            self.initialise_networks(self.model_path+'/params.pkl')
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/params.pkl'))
    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r']  # 训练时只需要自己的reward
        o, g, u, o_next,g_next= [], [], [],[],[]  # 用来装每个agent经验中的各项
        
        o.append(transitions['o'])
        g.append(transitions['g'])
        u.append(transitions['u'])
        o_next.append(transitions['o_next'])
        g_next.append(transitions['g_next'])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            #print(o_next[0].shape,g_next[0].shape)
            act_next = self.actor_target_network(o_next[0].to(device),g_next[0].to(device))
            u_next.append(act_next)
            q_next = self.critic_target_network(o_next[0].to(device), g_next[0].to(device), u_next[0]).detach()

            target_q = (r.unsqueeze(1).to(device) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o[0].to(device),g[0].to(device),u[0].to(device))
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[0]= self.actor_network(o[0].to(device),g[0].to(device))
        actor_loss = - self.critic_network(o[0].to(device),g[0].to(device),u[0]).mean()
        # if self.agent_id == 0:
        # print(actor_loss.)
        # print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # # update the network

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1


    def initialise_networks(self, path):
        
        checkpoint = torch.load(path) # load the torch data

        self.actor_network.load_state_dict(checkpoint['actor_params'])    # actor parameters
        self.critic_network.load_state_dict(checkpoint['critic_params'])    # critic parameters
        self.actor_optim.load_state_dict(checkpoint['actor_optim_params']) # actor optimiser state
        self.critic_optim.load_state_dict(checkpoint['critic_optim_params']) # critic optimiser state
        
        hard_update(self.actor_target_network, self.actor_network) # hard updates to initialise the targets
        hard_update(self.critic_target_network, self.critic_network)  # hard updates to initialise the targets

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        save_dict = {'actor_params' : self.actor_network.state_dict(),
                    'actor_optim_params': self.actor_optim.state_dict(),
                    'critic_params' : self.critic_network.state_dict(),
                    'critic_optim_params' : self.critic_optim.state_dict()}

        torch.save(save_dict, model_path + '/' + num + '_params.pkl')

