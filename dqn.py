import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, device):
        super(DeepQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(*input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def forward(self, state):
        return self.net(state)

    def load_from_network(self, network: nn.Module):
        self.load_state_dict(network.state_dict())
