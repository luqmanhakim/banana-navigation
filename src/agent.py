from .replaybuffer import *
from .model import *
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


class DqnAgent:
    def __init__(self, update_type="soft", state_size=37, action_size=4, hidden_sizes=[64, 32],
                 learn_every=3, transfer_every=5, lr=5e-4, double_dqn=False):
        """
        :param update_type: {'soft', 'hard'} hard update copies the local weights fully to the target weights
        :param state_size: state size of the env
        :param action_size: action size of the env
        :param hidden_sizes: list mirroring the neural network architecture of the agent
        :param learn_every: freqency of learning. higher int = longer to learn
        :param transfer_every: frequency of network weights transfer from local to the target. affects learning.
        :param lr: learning rate of the network optimizer
        :param double_dqn: boolean to use double dqn or not
        """

        self.double_dqn = double_dqn

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size

        # Build our local and target networks
        network_sizes = []
        network_sizes.append(state_size)
        network_sizes.extend(hidden_sizes)
        network_sizes.append(action_size)
        self.local_network = Q_Network(network_sizes).to(self.device)
        self.target_network = Q_Network(network_sizes).to(self.device)

        self.optimizer = optim.Adam(self.local_network.parameters(), lr=lr)

        # Init memory and size of sample taken randomly from it
        self.memory = ReplayBuffer(device=self.device)
        self.LEARNING_SAMPLE_SIZE = 10

        # Set learning and transfer update cycles and counters
        self.learn_every = learn_every
        self.transfer_every = transfer_every
        self.learn_counter = 0
        self.transfer_counter = 0

        # set the kind of transfer to be used
        self.update_type = update_type  # {"soft","hard"}
        # set transfer rate for soft-transfer
        self.TAU = 1e-3

        # set `concentration` of next target's Q-value to learn
        self.GAMMA = 0.99

    def step(self, state, action, reward, next_state, done):
        # keep in our reply buffer at every step
        self.memory.add(state, action, reward, next_state, done)

        # sample some experiences and learn every update step
        self.learn_counter = (self.learn_counter + 1) % self.learn_every
        if self.learn_counter == 0 and self.memory.size() >= self.LEARNING_SAMPLE_SIZE:
            sampled_experiences = self.memory.sample(self.LEARNING_SAMPLE_SIZE)
            self._learn(sampled_experiences)

            # after some time, transfer the weights from the Q-network to the Q-target
            self.transfer_counter = (self.transfer_counter + 1) % self.transfer_every
            if self.transfer_counter == 0:
                self._commit_learning()

    def act(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # turn off network learning mode
        self.local_network.eval()

        # temporarily set requires_grad flag to false
        with torch.no_grad():
            action_values = self.local_network(state)
            action = self._choose_action(action_values, eps)

        # set network back to learning mode
        self.local_network.train()

        return action

    def _choose_action(self, action_values, eps):
        # Epsilon-greedy action selection
        if random.random() > eps:
            # print("taking network action")
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # print("taking random action")
            return random.choice(np.arange(self.action_size))

    def _learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        if self.double_dqn:
            best_actions = self.local_network(next_states).detach().argmax(1).unsqueeze(1)
            # get actual Q-value from the actions taken
            Q_next_targets = self.target_network(next_states).detach().gather(1, best_actions)
        else:
            Q_next_targets = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (self.GAMMA * Q_next_targets * (1-dones))
        Q_actual = self.local_network(states).gather(1, actions)

        # calculate the loss
        loss = F.mse_loss(Q_actual, Q_targets)

        # minimize loss
        self.optimizer.zero_grad()
        loss.backward()

        # register the gradients
        self.optimizer.step()

    def _commit_learning(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
            if self.update_type == "soft":
                target_param.data.copy_((1.0-self.TAU)*target_param.data + self.TAU*local_param.data)
            if self.update_type == "hard":
                target_param.data.copy_(local_param.data)
