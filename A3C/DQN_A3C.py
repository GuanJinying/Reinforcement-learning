import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

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
        option_price = stats.norm.cdf(d1)*current_stock_price - stats.norm.cdf(d2)*self.initial_price*np.exp(-self.r*time_to_maturity)
        self.option_price = option_price
    
    # generate one step given the current_step
    def step(self):
        self.stock_price_generate()
        self.option_price_generate()
        self.current_time += 1
        
    def reset(self):
        self.stock_price = self.initial_price
        self.current_time = 0
        self.option_price_generate()
        
def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-5, betas=(0.93, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
        # state initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    def __init__(self, sigma, mu, initial_price, n, r):
        super(ActorCritic, self).__init__()
        self.price = price(sigma, mu, initial_price, n, r)
        
        self.a1 = nn.Linear(3, 50)
        self.value1 = nn.Linear(3, 50)

        self.b1 = nn.Linear(50, 50)
        self.value2 = nn.Linear(50, 50)
        self.mu = nn.Linear(50, 1)
        self.sigma = nn.Linear(50, 1)
        self.value = nn.Linear(50, 1)

        set_init([self.a1, self.value1, self.mu, self.sigma, self.value])

        self.rewards = []
        self.actions = []
        self.states = []
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
    def forward(self, state):
        a1 = torch.sigmoid(self.a1(state))
        value1 = torch.sigmoid(self.value1(state))
        b1 = torch.sigmoid(a1)
        mu = torch.tanh(self.mu(b1))
        sigma = F.softplus(self.sigma(b1)) + 0.001
        value2 = self.value2(value1)
        value = self.value(value2)
        return mu, sigma, value
        
    
    def calculate_loss(self, end):
        
        self.train()
        actions = torch.tensor(self.actions, dtype = torch.float)
        
        # Calculate R
        R = np.zeros(len(self.rewards))
        if end == 0:
            value = self.forward(torch.tensor(self.states[-1], dtype = torch.float))[2]
            R[-1] = value.detach().numpy()[0]
        else:
            R[-1] = 0

        for j in range(2, len(self.rewards) + 1):
            R[-j] = self.price.discount*R[-j + 1] + self.rewards[-j]
        total_reward = R[0]
        returns = torch.from_numpy(R)

        states = np.vstack(self.states)
        states = states.astype(np.float32)
        mu, sigma, values = self.forward(torch.from_numpy(states))

        # values = values.squeeze()
        td = returns - values
        critic_loss = td.pow(2)
        
        dist = torch.distributions.Normal(mu, sigma)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*td.detach()
        total_loss = (critic_loss + actor_loss).mean()
        
        return total_loss, total_reward
        
    def choose_action(self, state):
        self.training = False
        state = torch.tensor(state, dtype = torch.float)
        mu, sigma, v= self.forward(state)
        action_distribution = torch.distributions.Normal(mu.view(1, ).data, sigma.view(1, ).data)
        action = action_distribution.sample().numpy()
        return action[0]
    
    def get_reward(self, state, next_state, action):
        reward = 0
        pnl = (next_state[0]*action + next_state[1])*self.price.discount - (state[0]*action + state[1])
        reward = -abs(pnl)
        return reward
    
    def step(self, state):
        action = self.choose_action(state)
        self.price.step()
        next_state = [self.price.stock_price, self.price.option_price, self.price.current_time]
        reward = self.get_reward(state, next_state, action)
        return action, reward, next_state

        
class A3C_Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, sigma, mu, initial_price, n, r, training_time, reward_queue, global_episode, global_episode_reward, update_iter):
        super(A3C_Agent, self).__init__()
        self.local_actor_critic = ActorCritic(sigma, mu, initial_price, n, r)
        self.global_actor_critic = global_actor_critic
        self.optimizer = optimizer
        self.training_time = training_time
        self.reward_queue = reward_queue
        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.update_iter = update_iter
        
    def run(self):
        
        while self.global_episode.value < self.training_time:
            # reset the environment for local actor critic agent
            self.local_actor_critic.price.reset()
            state = [self.local_actor_critic.price.stock_price, self.local_actor_critic.price.option_price, self.local_actor_critic.price.current_time]
            self.local_actor_critic.clear_memory()
            
            while self.local_actor_critic.price.current_time < self.local_actor_critic.price.n:
                action, reward, next_state = self.local_actor_critic.step(state)
                self.local_actor_critic.remember(state, action, reward)
                
                if self.local_actor_critic.price.current_time % self.update_iter == 0 or self.local_actor_critic.price.current_time == self.local_actor_critic.price.n:
                    
                    end = 0
                    if self.local_actor_critic.price.current_time == self.local_actor_critic.price.n:
                        end = 1
                    # calculate local gradients and push local parameters to global parameters
                    loss, total_reward = self.local_actor_critic.calculate_loss(end)
                    self.optimizer.zero_grad()
                    loss.backward()
            
                    for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad

                    self.optimizer.step()

                    # pull the global parameters to local parameters
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())

                    if end == 1:
                        with self.global_episode.get_lock():
                            self.global_episode.value += 1
                        with self.global_episode_reward.get_lock():
                            if self.global_episode_reward.value == 0.:
                                self.global_episode_reward.value = total_reward
                            else:
                                self.global_episode_reward.value = self.global_episode_reward.value * 0.99 + total_reward * 0.01
                        self.reward_queue.put(total_reward)
                
                state = next_state

        self.reward_queue.put(None)


def A3C_DQN():
    # Initialize the parameter for both local actor critic agent and global actor critic agent
    lr = 5e-6
    sigma = 0.1
    mu = 1
    r = 0.02
    initial_price = 100
    n = 100
    training_time = 5000
    update_iter = 10
    # initialize global actor critic agent
    global_actor_critic = ActorCritic(sigma, mu, initial_price, n, r)
    global_actor_critic.share_memory()
    optimizer = SharedAdam(global_actor_critic.parameters(), lr = lr, betas = (0.95, 0.999))

    # set multiprocessing parameters
    global_episode = mp.Value('i', 0)
    global_episode_reward = mp.Value('d', 0.)
    reward_queue = mp.Queue()

    # Use the pytorch multiprocessing tool to execute parallel programming
    workers = [A3C_Agent(global_actor_critic, optimizer, sigma, mu, initial_price, n, r, training_time, reward_queue, global_episode, global_episode_reward, update_iter) for i in range(4)]
    [w.start() for w in workers]
        
    res = []                    
    while True:
        r = reward_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    
    Reward = pd.DataFrame({'reward':res})
    Reward.plot(title = 'Asychronous Advantage Actor-Critic Algorithm Total Reward')
    plt.show()

if __name__ == '__main__':
    A3C_DQN()
