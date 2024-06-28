import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = gym.make('ALE/Othello-v5')
state_size = (210, 160, 3) 
action_size = 10

batch_size = 64
n_episodes = 100
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
target_update = 10


class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(state_size[1], 8, 4), 4, 2), 3, 1)
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(state_size[0], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, action_size)
        self.init_weights()

    def conv2d_size_out(self, size, kernel_size, stride):
        return (size - kernel_size) // stride + 1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    


policy_net1 = DQNetwork(state_size, action_size).to(device)
target_net1 = DQNetwork(state_size, action_size).to(device)
target_net1.load_state_dict(policy_net1.state_dict())
target_net1.eval()

policy_net2 = DQNetwork(state_size, action_size).to(device)
target_net2 = DQNetwork(state_size, action_size).to(device)
target_net2.load_state_dict(policy_net2.state_dict())
target_net2.eval()

optimizer1 = optim.Adam(policy_net1.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(policy_net2.parameters(), lr=learning_rate)
criterion = nn.SmoothL1Loss() 



class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

memory1 = ReplayMemory(10000)
memory2 = ReplayMemory(10000)


def select_action(state, epsilon, policy_net):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            q_values = policy_net(state)
            return q_values.max(1)[1].item()
        


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))
    
    states = np.array(batch[0])
    states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)
    actions = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(batch[2]).to(device)
    next_states = np.array(batch[3])
    next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(device)
    dones = torch.FloatTensor(batch[4]).to(device)
    
    current_q_values = policy_net(states).gather(1, actions)
    max_next_q_values = target_net(next_states).max(1)[0].detach()
    expected_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
    
    loss = criterion(current_q_values, expected_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# Training loop for both agents
for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward1 = 0
    total_reward2 = 0
    epsilon1 = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
    epsilon2 = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
    
    for t in range(1000):
        # Agent 1's turn
        action1 = select_action(state, epsilon1, policy_net1)
        next_state, reward1, done, _, _ = env.step(action1)
        
        if done:
            if reward1 > 0:
                reward1 = 1  # Agent 1 wins
                reward2 = -1  # Agent 2 loses
            elif reward1 < 0:
                reward1 = -1  # Agent 1 loses
                reward2 = 1  # Agent 2 wins
            else:
                reward1 = 0  # Draw
                reward2 = 0  # Draw
        else:
            reward1 = reward1 / 10.0  # Normalize in-game rewards
        
        memory1.push((state, action1, reward1, next_state, done))
        state = next_state
        total_reward1 += reward1
        optimize_model(memory1, policy_net1, target_net1, optimizer1)
        
        if done:
            break
        
        # Agent 2's turn
        action2 = select_action(state, epsilon2, policy_net2)
        next_state, reward2, done, _, _ = env.step(action2)
        
        if done:
            if reward2 > 0:
                reward2 = 1  # Agent 2 wins
                reward1 = -1  # Agent 1 loses
            elif reward2 < 0:
                reward2 = -1  # Agent 2 loses
                reward1 = 1  # Agent 1 wins
            else:
                reward2 = 0  # Draw
                reward1 = 0  # Draw
        else:
            reward2 = reward2 / 10.0 
        
        memory2.push((state, action2, reward2, next_state, done))
        state = next_state
        total_reward2 += reward2
        optimize_model(memory2, policy_net2, target_net2, optimizer2)
        
        if done:
            break
    
    if episode % target_update == 0:
        target_net1.load_state_dict(policy_net1.state_dict())
        target_net2.load_state_dict(policy_net2.state_dict())
    
    print(f'Episode {episode}, Total Reward Agent 1: {total_reward1}, Total Reward Agent 2: {total_reward2}')
    
env.close()