import torch
import os
from pyrep.networks.model_attention import Attention_Actor, Attention_Critic
import numpy as np

MSELoss = torch.nn.MSELoss()

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
        self.actor_network = Attention_Actor(args, args.actor_hidden_dim)
        self.critic_network = Attention_Critic(args, args.critic_hidden_dim, args.critic_attend_heads)

        # build up the target network
        self.actor_target_network = Attention_Actor(args, args.actor_hidden_dim)
        self.critic_target_network = Attention_Critic(args, args.critic_hidden_dim, args.critic_attend_heads)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic, weight_decay=1e-3)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/params.pkl'):
            self.initialise_networks(self.model_path+'/params.pkl')
            print('Successfully loaded actor and critic networks')

        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics

        self.niter = 0

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def select_action(self, o, noise_rate, epsilon):
        us = []
       
        pi = self.actor_network(o) #n*[none,a_dim]
        # print('{} : {}'.format(self.name, pi))
        for i in range(self.args.n_agents):
            if np.random.uniform() < epsilon:
                u = np.random.uniform(-self.args.high_action, self.args.high_action, [pi[0].shape[0],self.args.action_shape[i]])
            else:
                u = pi[i].cpu().numpy() 
                noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
                # print("noise: ", np.random.randn(*u.shape))
                u += noise
                u = np.clip(u, -self.args.high_action, self.args.high_action)
            us.append(u)

        return us

    # update the network
    def train(self, transitions, logger, use_gpu):
        for key in transitions.keys():
            if use_gpu:
                transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).cuda()
            else:
                transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)

        o, u, o_next,r = [], [], [],[]  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
            r.append(transitions['r_%d' % agent_id])

        # calculate the target Q value function
       
        with torch.no_grad():
            # 得到下一个状态对应的动作
            u_next = self.actor_target_network(o_next)
            q_next = self.critic_target_network(o_next, u_next)
        
        q_value = self.critic_network(o, u, regularize=True, logger=logger,niter=self.niter)

        q_loss = 0
        
        for i, (nq), (q,regs) in zip(range(self.args.n_agents), q_next,
                                            q_value):
            target_q = (r[i].view(-1, 1) +
                        self.args.gamma * nq[0])
            # print(q,regs)
            q_loss += MSELoss(q, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention

        q_loss.backward()
        self.critic_network.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.critic_network.parameters(), 10 * self.args.n_agents)
        self.critic_optim.step()
        self.critic_optim.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        actor_loss = 0

        u_al = self.actor_network(o) #all actions from current policies
        u_co = [u for i in range(self.args.n_agents)]
        o_co = [o for i in range(self.args.n_agents)]

        for i in range(self.args.n_agents):
            u_co[i][i] = u_al[i]

        for i in range(self.args.n_agents):
            actor_loss += - self.critic_network(o_co[i], u_co[i],regularize=True)[i][0].mean() # n*[None,1]

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_network.scale_shared_grads()
        grad_norm1 = torch.nn.utils.clip_grad_norm(
            self.actor_network.parameters(), 10 * self.args.n_agents)
        self.actor_optim.step()

        if logger is not None:
            logger.add_scalar('losses/a_loss', actor_loss, self.niter)
            logger.add_scalar('grad_norms/aq', grad_norm1, self.niter)

        self.niter += 1

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def prep_training(self, device='gpu'):
        self.critic_network.train()
        self.critic_target_network.train()
        self.actor_network.train()
        self.actor_target_network.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()

        if not self.pol_dev == device:
            self.actor_network = fn(self.actor_network)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic_network = fn(self.critic_network)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            self.actor_target_network = fn(self.actor_target_network)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.critic_target_network = fn(self.critic_target_network)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        
        self.actor_network.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            self.actor_network = fn(self.actor_network)
            self.pol_dev = device

    def initialise_networks(self, path):
        
        checkpoint = torch.load(path) # load the torch data
        
        self.actor_network.load_state_dict(checkpoint['actor_params'])    # actor parameters
        self.actor_target_network.load_state_dict(checkpoint['actor_target_params']) 
        self.actor_optim.load_state_dict(checkpoint['actor_optim_params']) # actor optimiser state

        self.critic_network.load_state_dict(checkpoint['critic_params'])    # critic parameters
        self.critic_optim.load_state_dict(checkpoint['critic_optim_params']) # critic optimiser state
        self.critic_target_network.load_state_dict(checkpoint['critic_target_params']) 


    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        save_dict = {'actor_params' : [self.actor_network.state_dict()],
                    'actor_optim_params': [self.actor_optim.state_dict()],
                    'actor_target_params': [self.actor_target_network.state_dict()],
                    'critic_params' : self.critic_network.state_dict(),
                    'critic_optim_params' : self.critic_optim.state_dict(),
                    'critic_target_params' : self.critic_target_network.state_dict()}

        torch.save(save_dict, model_path + '/' + num + '_params.pkl')

