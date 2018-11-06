import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_Network(nn.Module):
    def __init__(self, network_sizes=[64,32], seed=42):
        """ Builds the neural network for agent training
        :param network_sizes: list of nodes for each layer
        :param seed: int for replicability of learning.
        """
        super(Q_Network, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.network = nn.ModuleList()
        for k in range(len(network_sizes)-1):
            self.network.append(nn.Linear(network_sizes[k], network_sizes[k+1]))

    def forward(self, state):
        x = state
        for fc in self.network[:-1]:
            x = F.relu(fc(x))
        output = F.softmax(self.network[-1](x), dim=1)
        return output

'''
class Duel_Network(nn.Module):
    def __init__(self, network_sizes=[64,32], seed=42):
        super(Duel_Network, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc_advantage = Q_Network(network_sizes)
        network_sizes.append(1)
        self.fc_value = Q_Network(network_sizes)

    def forward(self, state):
        advantage = self.fc_advantage(state)
        value = self.fc_value(state)
        q = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
        return q
'''