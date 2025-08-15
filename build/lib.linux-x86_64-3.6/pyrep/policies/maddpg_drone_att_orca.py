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
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor1(args,agent_id)
        self.critic_network = Critic1(args)

        # build up the target network
        self.actor_target_network = Actor1(args,agent_id)
        self.critic_target_network = Critic1(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

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
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
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

    #--------------------ORCA--------------------------------
    def ORCA_distance(self, vel, points):
        #input is array, vel is tensor
        point = vel
        line_point1 = points[:2]
        line_point2 = points[:2]+points[2:4]

        vec1 = line_point1-point
        vec2 = line_point2-point

        vec1_3d = torch.zeros(3)
        vec2_3d = torch.zeros(3)
        vec1_3d[0] = vec1[0]
        vec1_3d[1] = vec1[1]
        vec2_3d[0] = vec2[0]
        vec2_3d[1] = vec2[1]

        distance = torch.abs(torch.cross(vec1_3d,vec2_3d)[2])/torch.linalg.norm(line_point1-line_point2)

        return distance

    # update the network
    def train_ORCA(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        p_vel = transitions['prefer_vel_%d' % self.agent_id]
        plane = transitions['plane_%d' % self.agent_id]
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # use closest distance to half plane to optimize actor
        # self.actor_network(o[self.agent_id])
        loss = torch.zeros(plane.shape[0])
        
        for i in range(plane.shape[0]):
            d_sum = 0
            v1 = self.actor_network(o[self.agent_id][i,:])
            for j in range(plane[i,36].to(torch.int)):
                d_sum += self.ORCA_distance(v1,plane[i,j*4:j*4+4])               
            loss_local = d_sum + (v1-p_vel[i,:]).pow(2).sum()
            loss[i] = loss_local 
        
        actor_loss = loss.mean()
        # if self.agent_id == 0:
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

    # def save_model(self, train_step):
    #     num = str(train_step // self.args.save_rate)
    #     model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
    #     if not os.path.exists(model_path):
    #         os.makedirs(model_path)
    #     model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
    #     if not os.path.exists(model_path):
    #         os.makedirs(model_path)
    #     torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
    #     torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')
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


# class Centralized_MADDPG:
#     def __init__(self, args): 
#         self.args = args
#         self.train_step = 0

#         # create the network
#         self.actor_networks = [Actor(args, id) for id in range(args.n_agents)]
#         self.critic_network = Critic(args)

#         # build up the target network
#         self.actor_target_networks = [Actor(args, id) for id in range(args.n_agents)]
#         self.critic_target_network = Critic(args)

#         # load the weights into the target networks
#         for i, ac in enumerate(self.actor_target_networks):
#             ac.load_state_dict(self.actor_networks[i].state_dict())
#         self.critic_target_network.load_state_dict(self.critic_network.state_dict())

#         # create the optimizer
#         self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
#         self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

#         # create the dict for store the model
#         if not os.path.exists(self.args.save_dir):
#             os.mkdir(self.args.save_dir)
#         # path to save the model
#         self.model_path = self.args.save_dir + '/' + self.args.scenario_name
#         if not os.path.exists(self.model_path):
#             os.mkdir(self.model_path)
#         self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
#         if not os.path.exists(self.model_path):
#             os.mkdir(self.model_path)

#         # 加载模型
#         if os.path.exists(self.model_path + '/actor_params.pkl'):
#             self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
#             self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
#             print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
#                                                                           self.model_path + '/actor_params.pkl'))
#             print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
#                                                                            self.model_path + '/critic_params.pkl'))

#     # soft update
#     def _soft_update_target_network(self):
#         for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
#             target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

#         for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
#             target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

#     # update the network
#     def train(self, transitions, other_agents):
#         for key in transitions.keys():
#             transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
#         r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
#         o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
#         for agent_id in range(self.args.n_agents):
#             o.append(transitions['o_%d' % agent_id])
#             u.append(transitions['u_%d' % agent_id])
#             o_next.append(transitions['o_next_%d' % agent_id])

#         # calculate the target Q value function
#         u_next = []
#         with torch.no_grad():
#             # 得到下一个状态对应的动作
#             index = 0
#             for agent_id in range(self.args.n_agents):
#                 if agent_id == self.agent_id:
#                     u_next.append(self.actor_target_network(o_next[agent_id]))
#                 else:
#                     # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
#                     u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
#                     index += 1
#             q_next = self.critic_target_network(o_next, u_next).detach()

#             target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

#         # the q loss
#         q_value = self.critic_network(o, u)
#         critic_loss = (target_q - q_value).pow(2).mean()

#         # the actor loss
#         # 重新选择联合动作中当前agent的动作，其他agent的动作不变
#         u[self.agent_id] = self.actor_network(o[self.agent_id])
#         actor_loss = - self.critic_network(o, u).mean()
#         # if self.agent_id == 0:
#         #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
#         # # update the network
#         self.actor_optim.zero_grad()
#         actor_loss.backward()
#         self.actor_optim.step()
#         self.critic_optim.zero_grad()
#         critic_loss.backward()
#         self.critic_optim.step()

#         self._soft_update_target_network()
#         if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
#             self.save_model(self.train_step)
#         self.train_step += 1

#     def save_model(self, train_step):
#         num = str(train_step // self.args.save_rate)
#         model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
#         if not os.path.exists(model_path):
#             os.makedirs(model_path)
#         model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
#         if not os.path.exists(model_path):
#             os.makedirs(model_path)
#         torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
#         torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


