import torch
import os
from pyrep.networks.model_attention_shared_curriculum import Actor_attf as Actor1
from pyrep.networks.model_attention_shared_curriculum import Critic_attf as Critic1
# from pyrep.networks.actor_critic import Critic1
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
        self.actor_network = Actor1(args)
        self.critic_network = Critic1(args)

        # build up the target network
        self.actor_target_network = Actor1(args)
        self.critic_target_network = Critic1(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # # create the optimizer
        # self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        # self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir+ '/' + 'agent'
       
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if args.evaluate:
            model_path = self.args.save_dir + '/' + 'agent' 
            if os.path.exists(model_path + '/params.pkl'):
                self.initialise_networks(model_path + '/params.pkl')
                print('Evaluation: Agent successfully loaded actor_network: {}'.format(model_path + '/params.pkl'))
            else:
                print("cannot find any model")
        else:
            if args.stage == 2:
                if args.n_agents == 4:
                    model_path = "./" + args.scenario_name + "/stage1"+"/model_drone3"+ '/' + 'agent' 
                    if os.path.exists(model_path + '/params.pkl'):
                        self.initialise_networks(model_path + '/params.pkl')
                        print('Agent successfully loaded actor_network: {}'.format(model_path + '/params.pkl'))
                    else:
                        print("cannot find any model")
                
                if args.n_agents == 6:
                    model_path = "./" + args.scenario_name + "/stage2" + "/model_drone4"+ '/' + 'agent' 
                    if os.path.exists(model_path + '/params.pkl'):
                        self.initialise_networks(model_path + '/params.pkl')
                    else:
                        print("cannot find any model")

            if args.stage == 3:
                if args.n_agent == 4:
                    model_path = "./" + "navigation_curriculum" + "/stage2" + "/model_drone4"+ '/' + 'agent' 
                    if os.path.exists(model_path + '/params.pkl'):
                        self.initialise_networks(model_path + '/params.pkl')
                else:
                    print("stage3 must have four agents")

            if args.stage == 4:
                if args.n_agent == 6:
                    model_path = "./" + "navigation_curriculum" + "/stage3" + "/model_drone4"+ '/' + 'agent' 
                    if os.path.exists(model_path + '/params.pkl'):
                        self.initialise_networks(model_path + '/params.pkl')
                else:
                    print("stage4 must have six agents")

        # create the optimizer
        # if args.stage == 2:

        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
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
        o, u, o_next, r, done = [], [], [] ,[],[] # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
            r.append(transitions['r_%d' % agent_id] )
            done.append(transitions['done_%d' % agent_id])

        # calculate the target Q value function
     
        with torch.no_grad():
            u_next = self.actor_target_network(o_next)
           
            q_next_l = [tt.detach() for tt in self.critic_target_network(o_next, u_next)] #n*[none,1]

            target_q_l = []
            for i in range(self.args.n_agents):
                # if self.args.stage == 2:
                target_q = (r[i].unsqueeze(1) + (1-done[i]).unsqueeze(1)*self.args.gamma * q_next_l[i]).detach()
                
                # target_q = (r[i].unsqueeze(1) + self.args.gamma * q_next_l[i]).detach()
                # target_q = (r[i].unsqueeze(1) + self.args.gamma * self.critic_target_network(o_next, u_next)).detach()
                
                target_q_l.append(target_q)

        # the q loss
        q_value_l = self.critic_network(o, u, logger=logger, iter_n=self.niter) 
        # q_value_l = self.critic_network(o, u) 

        critic_loss = 0
        actor_loss = 0
        u_cl = self.actor_network(o)

        for i in range(self.args.n_agents):
            critic_loss += (target_q_l[i] - q_value_l[i]).pow(2).mean()
            # critic_loss += (target_q_l[i] - self.critic_network(o, u)).pow(2).mean() 

            u_n = u.copy()
            u_n[i] = u_cl[i]
            actor_loss += - self.critic_network(o, u_n)[i].mean()
            # actor_loss += - self.critic_network(o, u_n).mean()
       
        # # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_network.scale_shared_grads()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_network.scale_shared_grads()
        self.critic_optim.step()

        if logger is not None:
            logger.add_scalar('Agent/losses/q_loss', critic_loss, self.niter)
       
            logger.add_scalar('Agent/losses/a_loss', actor_loss, self.niter)

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
        # self.actor_optim.load_state_dict(checkpoint['actor_optim_params']) # actor optimiser state
        # self.critic_optim.load_state_dict(checkpoint['critic_optim_params']) # critic optimiser state

        self.actor_target_network.load_state_dict(checkpoint['actor_target_params']) 
        self.critic_target_network.load_state_dict(checkpoint['critic_target_params'])

    def freeze_networks(self):
        self.actor_network.freeze_layers()
        self.critic_network.freeze_layers()

        self.actor_target_network.freeze_layers()
        self.critic_target_network.freeze_layers()

        
    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
       
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        save_dict = {'actor_params' : self.actor_network.state_dict(),
                    'actor_optim_params': self.actor_optim.state_dict(),
                    'actor_target_params': self.actor_target_network.state_dict(),
                    'critic_params' : self.critic_network.state_dict(),
                    'critic_target_params' : self.critic_target_network.state_dict(),
                    'critic_optim_params' : self.critic_optim.state_dict()}

        torch.save(save_dict, self.model_path + '/' + num + '_params.pkl')





    



