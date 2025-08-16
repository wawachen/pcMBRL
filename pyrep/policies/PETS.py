from torch import nn
from pyrep.networks.neural_nets import PtModel
import os
from time import localtime, strftime
import torch

TORCH_DEVICE = torch.device('cpu')

class PETS_model(nn.Module):
    def __init__(self, ensemble_size, model_in, model_out, load_model=False, num_a=3, robot="uav"):
        super(PETS_model,self).__init__()
        
        # Choose model based on robot type
        if robot == "turtlebot":
            self.net = PtModel(ensemble_size, model_in, model_out * 2).to(TORCH_DEVICE)
            print(f"Using PtModel1 (Turtlebot) with action constraints")
        else:
            self.net = PtModel(ensemble_size, model_in, model_out * 2).to(TORCH_DEVICE)
            print(f"Using PtModel (UAV) without action constraints")
            
        self.robot = robot
        if self.robot == "turtlebot":
            self.optim = torch.optim.Adam(self.net.parameters(), lr=0.0001) #0.001
        if self.robot == "uav":
            self.optim = torch.optim.Adam(self.net.parameters(), lr=0.0001) #0.001

        self.model_path = os.path.join("/home/xlab/MARL_transport/PETS_model_3d",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
        os.makedirs(self.model_path, exist_ok=True)

        self.num_a = num_a

        if load_model:
            load_path = "/home/xlab/MARL_transport/PETS_model_3d"
            if self.num_a == 3:
                if self.robot == "uav":
                    if os.path.exists(load_path + '/3_params.pkl'):
                        self.initialise_networks(load_path+'/3_params.pkl')
                        print('Agent successfully loaded PETS_network: {}'.format(load_path + '/3_params.pkl'))
                    else:
                        print('Failed to load model from PETS')
                if self.robot == "turtlebot":
                    if os.path.exists(load_path + '/t3_params.pkl'):
                        self.initialise_networks(load_path+'/t3_params.pkl')
                        print('Agent successfully loaded PETS_network: {}'.format(load_path + '/t3_params.pkl'))
                    else:
                        print('Failed to load tmodel from PETS')
            if self.num_a == 4:
                if self.robot == "uav":
                    if os.path.exists(load_path + '/4_params.pkl'):
                        self.initialise_networks(load_path+'/4_params.pkl')
                        print('Agent successfully loaded PETS_network: {}'.format(load_path + '/4_params.pkl'))
                    else:
                        print('Failed to load model from PETS')
                if self.robot == "turtlebot":
                    if os.path.exists(load_path + '/t4_params.pkl'):
                        self.initialise_networks(load_path+'/t4_params.pkl')
                        print('Agent successfully loaded PETS_network: {}'.format(load_path + '/t4_params.pkl'))
                    else:
                        print('Failed to load tmodel from PETS')
            if self.num_a == 6:
                if self.robot == "uav":
                    if os.path.exists(load_path + '/6_params.pkl'):
                        self.initialise_networks(load_path+'/6_params.pkl')
                        print('Agent successfully loaded PETS_network: {}'.format(load_path + '/6_params.pkl'))
                    else:
                        print('Failed to load model from PETS')
                if self.robot == "turtlebot":
                    if os.path.exists(load_path + '/t6_params.pkl'):
                        self.initialise_networks(load_path+'/t6_params.pkl')
                        print('Agent successfully loaded PETS_network: {}'.format(load_path + '/t6_params.pkl'))
                    else:
                        print('Failed to load tmodel from PETS')

    def initialise_networks(self, path):
        
        checkpoint = torch.load(path) # load the torch data

        self.net.load_state_dict(checkpoint['PETS_params'])    # actor parameters
        self.optim.load_state_dict(checkpoint['PETS_optim_params']) # critic optimiser state
        
    def save_model(self, train_step):
        num = str(train_step)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        save_dict = {'PETS_params' : self.net.state_dict(),
                    'PETS_optim_params' : self.optim.state_dict()}

        torch.save(save_dict, self.model_path + '/' + num + '_params.pkl') 
