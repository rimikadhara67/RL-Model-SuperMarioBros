# agent_nn.py
import torch
import torch.nn as nn
import numpy as np

class AgentNN(nn.Module):
    def __init__(self, input_dims, num_actions, freeze=False):
        super(AgentNN, self).__init__()
        c, h, w = input_dims
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # Change input channels to 4
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_dims)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        if freeze:
            self.freeze()

        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print("Agent_nn.py: Device is ", self.device)
        self.to(self.device)

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, 4, *shape[1:]))  # Change input channels to 4
        return int(np.prod(o.size()))

    def freeze(self):
        for p in self.conv_layers.parameters():
            p.requires_grad = False
        for p in self.fc_layers.parameters():
            p.requires_grad = False

    def forward(self, x):
        conv_out = self.conv_layers(x).view(x.size()[0], -1)
        return self.fc_layers(conv_out)
