import random
from ctypes import *

from py_interface import *

import copy
import gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=float,
                    default=1.0, help='simulation mode')

class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('node', c_double),
        ('app', c_double),
        ('cbr', c_double),
        ('pdr', c_double),
        ('cf', c_double),
        ('bitrate', c_double),
        ('ql', c_double)
    ]

class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('pred', c_double)
    ]

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

class DQN:
    def __init__(self, state_dim, action_dim):
        self.eval_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.learn_step = 0
        self.batch_size = 32
        self.target_replace = 100
        self.memory_counter = 0
        self.memory_capacity = 2000
        self.memory = np.zeros((self.memory_capacity, 2 * state_dim + 2))
        self.optimizer = Adam(self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.MSELoss()
        self.stats = {'MSE Loss': [], 'Returns': []}  # Add stats dictionary

    def choose_action(self, x):
        x = torch.Tensor(x)
        if np.random.uniform() > 0.99 ** self.memory_counter:
            action_value = self.eval_net.forward(x)
            action = torch.argmax(action_value, 0).numpy()
        else:
            action = np.random.randint(0, 2)  # Choose between 0 and 1
        return action

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def learn(self):
        self.learn_step += 1
        if self.learn_step % self.target_replace == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        sample_indices = np.random.choice(self.memory_capacity, self.batch_size)
        samples = self.memory[sample_indices, :]
        s = torch.Tensor(samples[:, :3])
        a = torch.LongTensor(samples[:, 3:4])
        r = torch.Tensor(samples[:, 4:5])
        s_ = torch.Tensor(samples[:, 5:])
        q_eval = self.eval_net(s).gather(1, a)
        q_next = self.target_net(s_).detach()
        q_target = r + 0.8 * q_next.max(1, keepdim=True)[0]
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.stats['MSE Loss'].append(loss.item())  # Store MSE loss
        self.stats['Returns'].append(r.sum().item())  # Store total return
		
def compute_reward(prev_state, next_state, action, prev_actions):
    default_bonus = 5.0  # Bonus reward for taking action 1.0 by default
    condition_bonus = 10.0  # Bonus reward for following the conditions and taking action 0.0
    repeat_action_penalty = 10.0  # Penalty for repeating the same action more than twice
    total_reward = 0.0

    # Add bonus for taking action 1.0 by default
    if action == 1.0:
        total_reward = default_bonus

    # Add higher bonus for following the conditions and taking action 0.0
    if (next_state[0] >= 0.4 or (next_state[1] <= 0.5 and next_state[2] <= 0.2)) and action == 0.0:
        total_reward = condition_bonus
        
    # Add penalty for repeating the same action more than twice
    if len(prev_actions) >= 2 and prev_actions[-1] == action and prev_actions[-2] == action:
        total_reward -= repeat_action_penalty

    return total_reward

def plot_stats(stats, filename='stats_plot.png'):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        vals = [np.mean(vals[i-5:i+5]) for i in range(5, len(vals)-5)]
        if len(stats) > 1:
            ax[i].plot(range(len(vals)), vals)
            ax[i].set_title(key, size=18)
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
    plt.tight_layout()
    plt.savefig(filename)  # Save the plot to a file

# Initialize DQN
dqn = DQN(3, 2)

args = parser.parse_args()

# Evaluation mode
if args.mode == 3.0:
    # Load the trained model
    dqn.eval_net.load_state_dict(torch.load('trained_model.pth'))

    # Set the network in evaluation mode
    dqn.eval_net.eval()

ns3Settings = { 'numSeeds': 4, 'mode': args.mode}
mempool_key = 1234                                          # memory pool key, arbitrary integer large than 1000
#mem_size = 2048 
mem_size = 1024                                           # memory pool size in bytes
memblock_key = 2333                                         # memory block key, need to keep the same in the ns-3 script
exp = Experiment(mempool_key, mem_size, 'wifi-simple-adhoc-grid-ac-clean', '../')      # Set up the ns-3 environment

prev_state = [0.0, 0.0, 0.0]
# Initialize prev_actions as an empty list
prev_actions = []

try:
    for i in range(10):
        exp.reset()                                             # Reset the environment
        rl = Ns3AIRL(memblock_key, Env, Act)			  # Link the shared memory block with ns-3 script
        
        ns3Settings['numSeeds'] = 4 + i                    
		        
        pro = exp.run(setting=ns3Settings, show_output=True)    # Set and run the ns-3 script (sim.cc)
        while not rl.isFinish():
            with rl as data:
                if data == None:
                    break
                    
                node = data.env.node
                app = data.env.app
                cbr = data.env.cbr
                pdr = data.env.pdr
                cf = data.env.cf
                bitr = data.env.bitrate
                ql = data.env.ql
                
                if args.mode == 0.0:
                    # Send all
                    data.act.pred = 1.0
                    
                if args.mode == 1.0:
                
                    state = [cbr, pdr, cf]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr, pdr, cf]
                    #reward = compute_reward(prev_state, next_state, action)
                    #reward = compute_reward(cbr, pdr, cf, action)
                    
                    # Compute reward with previous actions
                    reward = compute_reward(prev_state, next_state, action, prev_actions)
                
                    # Update prev_actions
                    prev_actions.append(action)
                    if len(prev_actions) > 2:
                        prev_actions.pop(0)
                    
                    dqn.store_transition(state, action, reward, next_state)
                    dqn.learn()
                    prev_state = next_state
                
                if args.mode == 3.0:
                    state = [cbr, pdr, cf]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr, pdr, cf]
                    reward = compute_reward(prev_state, next_state, action, prev_actions)
                    prev_state = next_state
                                  
                if args.mode == 2.0:
                    # Random
                    data.act.pred = random.choice([0.0, 1.0])
                    
        pro.wait()                                              # Wait the ns-3 to stop
    if args.mode == 1.0:
        plot_stats(dqn.stats, 'stats_plot.png')
        torch.save(dqn.eval_net.state_dict(), 'trained_model.pth')
    
except Exception as e:
    print('Something wrong')
    print(e)
finally:
    del exp