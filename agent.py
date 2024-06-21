# agent.py
import torch
import torch.nn as nn
from agent_nn import AgentNN
import random

def rename_keys_in_state_dict(state_dict):
    renamed_state_dict = {}
    for key in state_dict.keys():
        new_key = key
        if key.startswith('network.0.0'):
            new_key = key.replace('network.0.0', 'conv_layers.0')
        elif key.startswith('network.0.2'):
            new_key = key.replace('network.0.2', 'conv_layers.2')
        elif key.startswith('network.0.4'):
            new_key = key.replace('network.0.4', 'conv_layers.4')
        elif key.startswith('network.2'):
            new_key = key.replace('network.2', 'fc_layers.0')
        elif key.startswith('network.4'):
            new_key = key.replace('network.4', 'fc_layers.2')
        renamed_state_dict[new_key] = state_dict[key]
    return renamed_state_dict

class Agent:
    def __init__(self, input_dims, num_actions):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print("Agent.py: Device is ", self.device)
        
        self.online_network = AgentNN(input_dims, num_actions).to(self.device)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True).to(self.device)
        
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()

        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995

        self.learn_step_counter = 0
        self.replay_buffer = []

    def choose_action(self, state):
        if torch.rand(1).item() > self.epsilon:
            state = state.to(self.device)
            with torch.no_grad():
                action = self.online_network(state).argmax().item()
        else:
            action = torch.randint(0, self.online_network.fc_layers[-1].out_features, (1,)).item()
        return action

    def store_in_memory(self, state, action, reward, new_state, done):
        self.replay_buffer.append((state, action, reward, new_state, done))
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

    def learn(self):
        if len(self.replay_buffer) < 1000:
            return

        self.optimizer.zero_grad()

        batch = random.sample(self.replay_buffer, 64)
        state, action, reward, new_state, done = zip(*batch)

        state = torch.cat(state).to(self.device)
        new_state = torch.cat(new_state).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)

        q_eval = self.online_network(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        q_next = self.target_network(new_state).max(1)[0]
        q_target = reward + (1 - done.float()) * 0.99 * q_next

        loss = self.loss(q_eval, q_target)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % 1000 == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        state_dict = rename_keys_in_state_dict(state_dict)
        self.online_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
