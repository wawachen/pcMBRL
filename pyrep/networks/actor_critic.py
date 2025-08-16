import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        #homogeneous thus action space is the same
        self.action_out1 = nn.Linear(64, 1) #angular velocity [-1,1]
        self.action_out2 = nn.Linear(64, 1) #linear velocity [0,1]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        act1 = torch.tanh(self.action_out1(x))
        act2 = torch.sigmoid(self.action_out2(x))

        actions = torch.cat((act1,act2),dim=-1)

        return actions

# define the centralized actor network
# class Actor1(nn.Module):
#     def __init__(self, args):
#         super(Actor, self).__init__()
#         self.max_action = args.high_action
#         self.fc1 = nn.Linear(args.obs_shape, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 64)
#         self.action_out = nn.Linear(64, args.action_shape[agent_id])

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         actions = self.max_action * torch.tanh(self.action_out(x))

#         return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

        # self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.q_out = nn.Linear(128, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        # for i in range(len(action)):
        #     action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value



# define the actor network
class Actor1(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor1, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


# define the actor network
class Actor_ddpg(nn.Module):
    def __init__(self, args):
        super(Actor_ddpg, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[0])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class Critic1(nn.Module):
    def __init__(self, args):
        super(Critic1, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

        # self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.q_out = nn.Linear(128, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value

# for image input of actor
#now is for single agent
class Actor_im(nn.Module):
    """image plus goal """
    def __init__(self, args):
        super(Actor_im, self).__init__()
        self.max_action = args.high_action
        self.args = args
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2) #optical flow is 2 dimension
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        self.linear_input_size = 32*10*61
        self.act_fc1 = nn.Linear(self.linear_input_size, 256)
        self.act_fc2 =  nn.Linear(256+2, 128)
        self.act_fc3 = nn.Linear(128, 2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, goal):
        #print("actor",x.shape,goal.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)

        fc = F.relu(self.act_fc1(x.view(x.size(0),self.linear_input_size)))
        # print(fc.shape,goal.shape)
        fc1 = torch.cat((fc, goal), dim=-1)
        # print(fc1.shape)
        fc2 = F.relu(self.act_fc2(fc1))
        # print(self.act_fc3(fc2))
        out = self.max_action * torch.tanh(self.act_fc3(fc2))
        #we want output both action and the latent layer for critic
        #print("actor",out.shape)
        return out

#can be derived from Kinect camera
class Critic_im(nn.Module):
    def __init__(self, args):
        super(Critic_im, self).__init__()
        self.max_action = args.high_action
        #---------------------------------------------
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2) #optical flow is 2 dimension
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        self.linear_input_size = 32*10*61
        self.act_fc1 = nn.Linear(self.linear_input_size, 256)
        #---------------------------------------------
        self.fc1 = nn.Linear(256+2+2, 64) #input is the latent layer of actor
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, goal, action):
        action /= self.max_action
        #print("critic",state.shape,goal.shape,action.shape)
        #-----------------
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        fc = F.relu(self.act_fc1(x.view(x.size(0),self.linear_input_size)))
        #-----------------

        x = torch.cat([fc, goal, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        #print("critic",q_value.shape)
        return q_value

class Critic_ddpg(nn.Module):
    def __init__(self, args):
        super(Critic_ddpg, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[0] + args.action_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

        # self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.q_out = nn.Linear(128, 1)

    def forward(self, state, action):
        action = action/self.max_action

        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
 

# define the actor network
class Actor_t(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor_t, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape_t[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        #homogeneous thus action space is the same
        self.action_out1 = nn.Linear(64, 1) #angular velocity [-1,1]
        self.action_out2 = nn.Linear(64, 1) #linear velocity [0,1]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        act1 = torch.tanh(self.action_out1(x))
        act2 = torch.sigmoid(self.action_out2(x))

        actions = torch.cat((act1,act2),dim=-1)

        return actions

# define the centralized actor network
# class Actor1(nn.Module):
#     def __init__(self, args):
#         super(Actor, self).__init__()
#         self.max_action = args.high_action
#         self.fc1 = nn.Linear(args.obs_shape, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 64)
#         self.action_out = nn.Linear(64, args.action_shape[agent_id])

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         actions = self.max_action * torch.tanh(self.action_out(x))

#         return actions


class Critic_t(nn.Module):
    def __init__(self, args):
        super(Critic_t, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_shape_t) + sum(args.action_shape_t), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

        # self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.q_out = nn.Linear(128, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        # for i in range(len(action)):
        #     action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value