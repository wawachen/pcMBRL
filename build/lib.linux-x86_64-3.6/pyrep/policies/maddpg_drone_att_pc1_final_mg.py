import torch
import os
from pyrep.networks.model_attention_shared import Actor_attf as Actor1
from pyrep.networks.model_attention_shared import Critic_attf as Critic1
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
    def __init__(self, args, rank):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.train_step = 0
        self.rank = rank

        # create the network
        self.actor_network = Actor1(args)
        self.critic_network = Critic1(args)

        # build up the target network
        self.actor_target_network = Actor1(args)
        self.critic_target_network = Critic1(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        # if args.stage == 2 or args.stage == 4:
        #     if not os.path.exists(self.args.load_dir):
        #         print("cannot find load path")
        #         assert(1==0)
        #     # path to save the model
        #     self.load_path = self.args.load_dir
        #     self.load_path = self.load_path + '/' + 'agent_%d' % agent_id
        #     if not os.path.exists(self.load_path):
        #         print("cannot find load path")
        #         assert(1==0)

        # if args.stage == 3:
        #     self.load_path1 = self.args.load_dir1
        #     self.load_path2 = self.args.load_dir2
        #     # path to save the model
        #     if (not os.path.exists(self.load_path1)) or (not os.path.exists(self.load_path2)):
        #         print("cannot find load paths")
        #         assert(1==0)

        #     length = (args.selection_num*(args.selection_num+1)/2)/2
            
        #     if agent_id<length:
        #         self.load_path = self.load_path1 + '/' + 'agent_%d' % agent_id
        #     else:
        #         self.load_path = self.load_path2 + '/' + 'agent_%d' % (agent_id-length)
        
        # if args.stage > 1:
        #     if args.stage == 2:
        #         with open('{}_number1.npy'.format(self.rank), 'rb') as f:
        #             load_num = np.load(f)
        #     if args.stage == 3:
        #         if agent_id<length:
        #             with open('{}_number1.npy'.format(args.index_r[0]), 'rb') as f:
        #                 load_num = np.load(f)
        #         else:
        #             with open('{}_number1.npy'.format(args.index_r[1]), 'rb') as f:
        #                 load_num = np.load(f)
        #     if args.stage == 4:
        #         with open('{}_number3.npy'.format(self.rank), 'rb') as f:
        #             load_num = np.load(f)

        #     if os.path.exists(self.load_path + '/' + str(load_num) +'_params.pkl'):
        #         self.initialise_networks(self.load_path + '/' + str(load_num)+'_params.pkl')
        #         print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id, self.load_path + '/'+ str(load_num) + '_params.pkl'))
        #     else:
        #         print("cannot find load final path")
        #         assert(1==0)
        
        # self.pol_dev = 'cpu'  # device for policies
        # self.critic_dev = 'cpu'  # device for critics
        # self.trgt_pol_dev = 'cpu'  # device for target policies
        # self.trgt_critic_dev = 'cpu'  # device for target critics

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir
       
        self.model_path = self.model_path + '/' + 'agent'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/params.pkl'):
            self.initialise_networks(self.model_path+'/params.pkl')
            print('Agent successfully loaded actor_network: {}'.format(self.model_path + '/params.pkl'))
            
        self.niter = 0
                                                                    
    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)


    def select_actions(self, o, noise_rate, epsilon,use_gpu,logger):

        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(1)
        pis = self.actor_network(inputs,logger=logger, iter_n=self.niter) #n*[none, adim]

        u_list = []

        for i in range(self.args.n_agents):
            if np.random.uniform() < epsilon:
                u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[i])
            else:
                pi = pis[i].squeeze(0)
                # print('{} : {}'.format(self.name, pi))
                u = pi.cpu().numpy()
                noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
                # print("noise: ", np.random.randn(*u.shape))
                u += noise
                u = np.clip(u, -self.args.high_action, self.args.high_action)
            u_list.append(u)
    
        return u_list

    # update the network
    def train(self, transitions, logger, use_gpu):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        
        # r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward [none,]
        o, u, o_next, r = [], [], [] ,[] # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
            r.append(transitions['r_%d' % agent_id] )

        # calculate the target Q value function
     
        with torch.no_grad():
            u_next = self.actor_target_network(o_next)
            # # 得到下一个状态对应的动作
            # index = 0
            # for agent_id in range(self.args.n_agents):
            #     if agent_id == self.agent_id:
            #         u_next.append(self.actor_target_network(o_next[agent_id]))
            #     else:
            #         # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
            #         u_next.append(other_agents[index].actor_target_network(o_next[agent_id])) #n*[none, adim]
            #         index += 1
            q_next_l = [tt.detach() for tt in self.critic_target_network(o_next, u_next)] #n*[none,1]

            target_q_l = []
            for i in range(self.args.n_agents):
                target_q = (r[i].unsqueeze(1) + self.args.gamma * q_next_l[i]).detach()
                target_q_l.append(target_q)

        # the q loss
        q_value_l = self.critic_network(o, u, logger=logger, iter_n=self.niter) 

        critic_loss = 0
        actor_loss = 0
        u_cl = self.actor_network(o)

        for i in range(self.args.n_agents):
            critic_loss += (target_q_l[i] - q_value_l[i]).pow(2).mean()

            # if logger is not None:
            #     logger.add_scalar('Agent%d/losses/q_loss'%self.agent_id, critic_loss, self.niter)

            # the actor loss
            # 重新选择联合动作中当前agent的动作，其他agent的动作不变
            u_n = u.copy()
            u_n[i] = u_cl[i]
            actor_loss += - self.critic_network(o, u_n)[i].mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_network.scale_shared_grads()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_network.scale_shared_grads()
        self.critic_optim.step()

        # if logger is not None:
        #     logger.add_scalar('Agent%d/losses/a_loss'%self.agent_id, actor_loss, self.niter)

        self.niter += 1
        # print(self.niter)

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def prep_training(self, device='gpu'):
        self.critic_network.train()
        self.critic_target_network.train()
        self.actor_network.train()
        self.actor_target_network.train()

        # if device == 'gpu':
        #     fn = lambda x: x.cuda()
        # else:
        #     fn = lambda x: x.cpu()

        # if not self.pol_dev == device:
        #     self.actor_network = fn(self.actor_network)
        #     self.pol_dev = device
        # if not self.critic_dev == device:
        #     self.critic_network = fn(self.critic_network)
        #     self.critic_dev = device
        # if not self.trgt_pol_dev == device:
        #     self.actor_target_network = fn(self.actor_target_network)
        #     self.trgt_pol_dev = device
        # if not self.trgt_critic_dev == device:
        #     self.critic_target_network = fn(self.critic_target_network)
        #     self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        
        self.actor_network.eval()
        # if device == 'gpu':
        #     fn = lambda x: x.cuda()
        # else:
        #     fn = lambda x: x.cpu()
        # # only need main policy for rollouts
        # if not self.pol_dev == device:
        #     self.actor_network = fn(self.actor_network)
        #     self.pol_dev = device

    def initialise_networks(self, path):
        
        checkpoint = torch.load(path) # load the torch data

        self.actor_network.load_state_dict(checkpoint['actor_params'])    # actor parameters
        self.critic_network.load_state_dict(checkpoint['critic_params'])    # critic parameters
        self.actor_optim.load_state_dict(checkpoint['actor_optim_params']) # actor optimiser state
        self.critic_optim.load_state_dict(checkpoint['critic_optim_params']) # critic optimiser state

        self.actor_target_network.load_state_dict(checkpoint['actor_target_params'][0]) 
        self.critic_target_network.load_state_dict(checkpoint['critic_target_params'])
        
    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = self.args.save_dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        save_dict = {'actor_params' : self.actor_network.state_dict(),
                    'actor_optim_params': self.actor_optim.state_dict(),
                    'actor_target_params': [self.actor_target_network.state_dict()],
                    'critic_params' : self.critic_network.state_dict(),
                    'critic_target_params' : self.critic_target_network.state_dict(),
                    'critic_optim_params' : self.critic_optim.state_dict()}

        torch.save(save_dict, model_path + '/' + num + '_params.pkl')

        # if self.args.stage == 1:
        #     # print('{}_number1.npy'.format(self.rank))
        #     with open('{}_number1.npy'.format(self.rank), 'wb') as f:
        #         np.save(f, num)
        # if self.args.stage == 2:
        #     with open('{}_number2.npy'.format(self.rank), 'wb') as f:
        #         np.save(f, num)
        # if self.args.stage == 3:
        #     with open('{}_number3.npy'.format(self.rank), 'wb') as f:
        #         np.save(f, num)
        # if self.args.stage == 4:
        #     with open('{}_number4.npy'.format(self.rank), 'wb') as f:
        #         np.save(f, num)




    



