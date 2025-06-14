import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import ReplayBuffer

class DQN:
    def __init__(
        self,
        state_size,
        action_size,
        device,
        learning_rate,
        gamma,
        replay_buffer_capacity,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.qnet = QNet(state_size, action_size).to(device)
        self.target_qnet = QNet(state_size, action_size).to(device)
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.remember((state, action, reward, next_state, done))

    def act(self, state):
        self.qnet.eval()

        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(dim=0)
        with torch.no_grad():
            q_values = self.qnet(state_tensor)
        action = torch.argmax(q_values).item()

        return action

    def replay(self, batch_size):
        self.qnet.train()
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        q_values = self.qnet(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_qnet(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load(self, fname):
        self.qnet.load_state_dict(torch.load(fname, map_location=self.device))
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def save(self, fname):
        torch.save(self.qnet.state_dict(), fname)


class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
