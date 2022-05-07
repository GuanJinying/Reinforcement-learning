from multiprocessing.reduction import steal_handle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from copy import deepcopy

import torch
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

# set up the environment
class price():
    def __init__(self, sigma, mu, initial_price, n, r):
        self.sigma = sigma
        self.mu = mu
        self.initial_price = initial_price
        self.deltaT = 1/n
        self.n = n
        self.r = r
        self.stock_price = initial_price
        self.option_price = 0
        self.current_time = 0
        self.discount = np.exp(-r*self.deltaT)
        
    # generate a stock price given the current stock price
    def stock_price_generate(self):
            
        current_stock_price = self.stock_price
        rand = np.random.normal(0,1)
        next_stock_price = current_stock_price*np.exp((self.mu - self.sigma**2/2)*self.deltaT + self.sigma*rand*np.sqrt(self.deltaT))
        self.stock_price = next_stock_price
        
    # generate the option price
    def option_price_generate(self):
        
        current_stock_price = self.stock_price
        time_to_maturity = 1 - self.deltaT*self.current_time
        d1 = (np.log(current_stock_price/self.initial_price) + (self.mu + self.sigma**2/2)*time_to_maturity)/(self.sigma*np.sqrt(time_to_maturity))
        d2 = d1 - self.sigma*np.sqrt(time_to_maturity)
        delta = stats.norm.cdf(d1)
        option_price = delta*current_stock_price - stats.norm.cdf(d2)*self.initial_price*np.exp(-self.r*time_to_maturity)
        self.option_price = option_price
        return delta
    
    # generate one step given the current_step
    def step(self):
        self.stock_price_generate()
        delta = self.option_price_generate()
        self.current_time += 1
        return delta
        
    def reset(self):
        self.stock_price = self.initial_price
        self.current_time = 0
        delta = self.option_price_generate()
        return delta

    

# implement the experience replay buffer
class replay_buffer():
    def __init__(self, max_memory_space):
        self.max_memory_space = max_memory_space
        self.memory_count = 0
        self.state_memory = np.zeros((max_memory_space, 3))
        self.next_state_memory = np.zeros((max_memory_space, 3))
        self.action_memory = np.zeros(max_memory_space)
        self.reward_memory = np.zeros(max_memory_space)
        self.is_terminate_memory = np.zeros(max_memory_space)

    def add_memory(self, state, action, reward, next_state, is_terminate):
        update_index = self.memory_count

        # randomly choose one memory in the reply buffer to replace to make sure the independence 
        if self.memory_count >= self.max_memory_space:
            update_index = np.random.randint(0, self.max_memory_space)
        self.state_memory[update_index] = state
        self.action_memory[update_index] = action
        self.reward_memory[update_index] = reward
        self.next_state_memory[update_index] = next_state
        self.is_terminate_memory[update_index] = is_terminate
        if self.memory_count < self.max_memory_space:
            self.memory_count += 1

    def sample_buffer(self, num_of_sample):
        batch_index = np.random.choice(self.memory_count, size = num_of_sample, replace = False)
        batch_state = self.state_memory[batch_index]
        batch_action = self.action_memory[batch_index]
        batch_reward = self.reward_memory[batch_index]
        batch_next_state = self.next_state_memory[batch_index]
        batch_is_terminate = self.is_terminate_memory[batch_index]

        return dict(state = batch_state, action = batch_action, reward = batch_reward, next_state = batch_next_state, is_terminate = batch_is_terminate)

# add noice to the deterministic action value
class add_OU_Noice():
    def __init__(self, mu, theta, sigma, deltaT):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.deltaT = deltaT
        self.reset()

    def reset(self):
        self.state = self.mu

    def noice(self):
        x = self.state
        dx = self.theta*(self.mu - x)*self.deltaT + self.sigma*np.random.normal(0,1)*np.sqrt(self.deltaT)
        self.state = dx + x
        return self.state

# initialize the weight and bias of the network
def set_init(layers, init_weight):
    for layer in layers:
        nn.init.uniform_(layer.weight.data, -init_weight, init_weight)
        nn.init.uniform_(layer.bias.data, -init_weight, init_weight)

class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.first_layer = nn.Linear(3, 10)
        self.second_layer = nn.Linear(10, 10)
        self.output_layer = nn.Linear(10, 1)
        init_weight = 1e-3
        set_init([self.first_layer, self.second_layer, self.output_layer], init_weight)

    def forward(self, state):
        hidden1 = torch.sigmoid(self.first_layer(state))
        hidden2 = torch.sigmoid(self.second_layer(hidden1))
        action = torch.tanh(self.output_layer(hidden2))
        return action

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.first_layer = nn.Linear(4, 10)
        self.second_layer = nn.Linear(10, 10)
        self.out_layer = nn.Linear(10, 1)

        init_weight = 1e-3
        set_init([self.first_layer, self.second_layer, self.out_layer], init_weight)

    def forward(self, state, action):
        x = torch.cat((state, action), dim = -1)
        
        hidden1 = torch.sigmoid(self.first_layer(x))
        hidden2 = torch.sigmoid(self.second_layer(hidden1))
        value = self.out_layer(hidden2)

        return value

class DDPG_agent():
    def __init__(self, sigma, mu, initial_price, n, r, max_memory_size, batch_size, ou_theta, ou_sigma, lr, training_time, pho):
        self.training_time = training_time
        self.pho = pho
        self.price = price(sigma, mu, initial_price, n, r)
        self.delta = self.price.reset()
        self.replay_buffer = replay_buffer(max_memory_size)
        self.batch_size = batch_size
        self.add_OU_noice = add_OU_Noice(0, ou_theta, ou_sigma, self.price.deltaT)

        self.actor_local = ActorNetwork()
        self.actor_target = ActorNetwork()
        self.actor_target.load_state_dict(self.actor_local.state_dict())

        self.critic_local = CriticNetwork()
        self.critic_target = CriticNetwork()
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr = lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr = lr)

    def choose_action(self, state):
        self.training = False
        state = torch.tensor(state, dtype = torch.float)
        action = self.actor_local.forward(state).detach().numpy()[0]
        noice = self.add_OU_noice.noice()
        action = np.clip(noice + action, -1, 1)
        return action

    def get_reward(self, state, next_state, action):
        reward = 0
        pnl = (next_state[0]*action + next_state[1])*self.price.discount - (state[0]*action + state[1])
        reward = -abs(pnl)
        return reward

    def step(self, state):
        action = self.choose_action(state)
        delta = self.price.step()
        next_state = np.array([self.price.stock_price, self.price.option_price, self.price.current_time])
        reward = self.get_reward(state, next_state, action)
        is_terminate = 0
        if self.price.current_time == self.price.n - 1:
            is_terminate = 1
        state = np.array(state)
        self.replay_buffer.add_memory(state, action, reward, next_state, is_terminate)
        return action, reward, next_state, delta

    def update(self, sample_number):
        memory_buffer_sample = self.replay_buffer.sample_buffer(sample_number)
        states = torch.tensor(memory_buffer_sample['state'], dtype = torch.float)
        actions = torch.tensor(memory_buffer_sample['action'].reshape(-1,1), dtype = torch.float)
        rewards = torch.tensor(memory_buffer_sample['reward'].reshape(-1, 1), dtype = torch.float)
        next_states = torch.tensor(memory_buffer_sample['next_state'], dtype = torch.float)
        is_terminates = torch.tensor(memory_buffer_sample['is_terminate'].reshape(-1, 1), dtype = torch.float)

        next_actions = self.actor_target.forward(next_states)
        next_values = self.critic_target.forward(next_states, next_actions)
        targets = rewards + self.price.discount*(1 - is_terminates)* next_values

        # update the local critic network value
        values = self.critic_local.forward(states, actions)
        value_loss = F.mse_loss(targets, values)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # update the local actor network value
        actor_loss = -self.critic_local.forward(states, self.actor_local(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update the target critic network
        for local_param, target_param in zip(self.critic_local.parameters(), self.critic_target.parameters()):
            target_param = (1 - self.pho)*local_param.data + self.pho*target_param.data

        # update the target actor network
        for local_param, target_param in zip(self.actor_local.parameters(), self.actor_target.parameters()):
            target_param = (1 - self.pho)*local_param.data + self.pho*target_param.data

    def train(self):
        episode_reward = []
        delta_episode_reward = []
        total_reward = []
        delta_total_reward = []
        state = [self.price.stock_price, self.price.option_price, self.price.current_time]
        for i in range(self.training_time):
            
            action, reward, next_state, delta = self.step(state)
            delta_reward = self.get_reward(state, next_state, -self.delta)
            state = next_state
            episode_reward.append(reward)
            delta_episode_reward.append(delta)
            self.delta = delta
            if self.price.current_time == self.price.n - 1:
                R = np.zeros(len(episode_reward))
                R[-1] = episode_reward[-1]
                for j in range(2, len(episode_reward) + 1):
                    R[-j] = self.price.discount*R[-j + 1] + episode_reward[-j]
                total_reward.append(R[0])
                delta_R = np.zeros(len(delta_episode_reward))
                delta_R[-1] = delta_episode_reward[-1]
                for j in range(2, len(delta_episode_reward) + 1):
                    delta_R[-j] = self.price.discount*delta_R[-j + 1] + delta_episode_reward[-j]
                delta_total_reward.append(delta_R[0])
                episode_reward = []
                delta_episode_reward = []
                self.delta = self.price.reset()
            if self.replay_buffer.memory_count >= self.replay_buffer.max_memory_space:
                self.update(self.batch_size)
        return total_reward, delta_reward


def DDPG():
    lr = 5e-4
    sigma = 0.1
    mu = 1
    r = 0.02
    initial_price = 100
    n = 50
    training_time = 10000
    max_memory_size = 3000
    batch_size = 300
    ou_theta = 0.1
    ou_sigma = 0.01
    pho = 1e-3
    ddpg = DDPG_agent(sigma, mu, initial_price, n, r, max_memory_size, batch_size, ou_theta, ou_sigma, lr, training_time, pho)
    total_reward, delta_total_reward = ddpg.train()
    performance_df = pd.DataFrame({'DDPG_reward': total_reward, 'delta_reward': delta_total_reward})
    performance_df.plot(title = 'Deterministic Deep Policy Gradient Algorithm Total Reward', xlabel = 'episode number')
    plt.show()

if __name__ == "__main__":
    DDPG()