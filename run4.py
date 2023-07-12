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
import csv

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
            nn.Linear(state_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

class DQN:
    def __init__(self, state_dim, action_dim):
        self.eval_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.learn_step = 0
        self.batch_size = 32
        self.observer_shape = state_dim
        self.target_replace = 100
        self.memory_counter = 0
        self.memory_capacity = 2000
        self.memory = np.zeros((self.memory_capacity, 2 * state_dim + 2))
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.MSELoss()
        self.stats = {'MSE Loss': [], 'Returns': [], 'Cum Returns': []}  # Add stats dictionary
        self.cumulative_return = 0  # Initialize cumulative return to zero
        
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
        sample = self.memory[sample_indices, :]
        s = torch.Tensor(sample[:, :self.observer_shape])
        a = torch.LongTensor(sample[:, self.observer_shape:self.observer_shape+1])
        r = torch.Tensor(sample[:, self.observer_shape+1:self.observer_shape+2])
        s_ = torch.Tensor(sample[:, self.observer_shape+2:])
        q_eval = self.eval_net(s).gather(1, a)
        q_next = self.target_net(s_).detach()
        q_target = r + 0.8 * q_next.max(1, keepdim=True)[0]
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.stats['MSE Loss'].append(loss.item())  # Store MSE loss
        self.stats['Returns'].append(r.sum().item())  # Store total returns
        self.cumulative_return += r.sum().item()  # Add current reward to cumulative return
        self.stats['Cum Returns'].append(self.cumulative_return)  # Store cumulative return

def compute_reward_1(prev_state, next_state, action, prev_actions):
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

def compute_reward_2(prev_state, next_state, action, prev_actions):
    default_bonus = 5.0  # Bonus reward for taking action 1.0 by default
    condition_bonus = 10.0  # Bonus reward for following the conditions and taking action 0.0
    repeat_action_penalty = 10.0  # Penalty for repeating the same action more than twice
    total_reward = 0.0

    # Add bonus for taking action 1.0 by default
    if action == 1.0:
        total_reward = default_bonus

    # Add higher bonus for following the conditions and taking action 0.0
    if next_state[0] <= 0.4  and action == 0.0:
        total_reward = condition_bonus
        
    # Add penalty for repeating the same action more than twice
    if len(prev_actions) >= 2 and prev_actions[-1] == action and prev_actions[-2] == action:
        total_reward -= repeat_action_penalty

    return total_reward

def compute_reward_3(prev_state, next_state, action, prev_actions):
    default_bonus = 5.0  # Bonus reward for taking action 1.0 by default
    condition_bonus = 10.0  # Bonus reward for following the conditions and taking action 0.0
    penalty = -10.0  # Penalty for not following the conditions

    # Add bonus for taking action 1.0 by default
    if action == 1.0 and next_state[0] >= 0.8:
        total_reward = default_bonus

    # Add higher bonus for following the conditions and taking action 0.0
    elif action == 0.0 and next_state[0] < 0.8:
        total_reward = condition_bonus

    # Encourage the agent to switch back to action 1.0 when pdr is greater than 0.9
    elif action == 1.0 and next_state[0] > 0.9:
        total_reward = condition_bonus + default_bonus

    # Add penalty for taking action 0.0 when pdr >= 0.9
    elif action == 0.0 and next_state[0] >= 0.9:
        total_reward = penalty

    # Add penalty for taking action 1.0 when pdr <= 0.8
    elif action == 1.0 and next_state[0] <= 0.8:
        total_reward = penalty

    else:
        total_reward = -10.0  # No reward if the conditions are not met

    return total_reward
    
def compute_reward_4(prev_state, next_state, action, prev_actions):
    
    if next_state[0] >= 0.9:
        total_reward = next_state[0] * action * 10

    if next_state[0] < 0.9:
        total_reward = next_state[0] * (1 - action) * 10
    
    return total_reward

def compute_reward_5(cbr, pdr, action):
    # Induce action 1
    if cbr < 0.10 and pdr >= 0.9:
        total_reward = action * ((1 - cbr) + 2 * pdr) * 10 
    elif cbr < 0.10 and pdr < 0.9:
        total_reward = action * ((1 - cbr) + (1 - 2 * pdr)) * 10
    # Induce action 0
    elif cbr >= 0.10 and pdr >= 0.9:
        total_reward = (1 - action) * (2 * cbr + pdr) * 10
    elif cbr >= 0.10 and pdr < 0.9:
        total_reward = (1 - action) * (2 * cbr + (1 - pdr)) * 10

    return total_reward

def compute_reward_6(cbr, action):
    
    if cbr < 0.1:
        total_reward = (1-cbr) * action #* 10

    if cbr >= 0.1:
        total_reward = (1-cbr) * (1 - action) #* 10
    
    return total_reward

def compute_reward_7(nreps, cbr, action):

    # Induce action 1
    if nreps <= 0.5 and cbr < 0.1:
        total_reward = action * ((1 - nreps) + (1 - cbr)) * 10

    if nreps <= 0.5 and cbr >= 0.1:
        total_reward = action * ((1 - nreps) + cbr) * 10

    # Induce action 0
    if nreps > 0.5 and cbr < 0.1:
        total_reward = (1 - action) * (nreps + (1 - cbr)) * 10

    if nreps > 0.5 and cbr >= 0.1:
        total_reward = (1 - action) * (nreps + cbr) * 10
    
    return total_reward
    
def compute_reward_8(nreps, cbr, action):

    # Induce action 1
    if cbr < 0.1 and nreps <= 0.5 :
        total_reward = action * ((1 - nreps) + (1 - cbr)) * 10
    # Induce action 0
    if cbr >= 0.1 and nreps > 0.5 :
        total_reward = (1 - action) * (nreps + cbr) * 10
    else:
        total_reward = 0;
    
    return total_reward
    
def compute_reward_9(cbr, pdr, action):
    
    if cbr < 0.1:
        total_reward = ((1-cbr) * action) + (pdr * action)

    if cbr >= 0.1:
        total_reward = (cbr * (1 - action)) + (pdr * (1 - action))
    
    return total_reward

def plot_stats(stats, filename='stats_plot.png'):
    rows = len(stats) 
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 9))

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
	
def save_stats_to_tsv(stats, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        # Get the keys (column names)
        keys = list(stats.keys())
        # Get the length of the lists (number of rows)
        num_rows = len(stats[keys[0]])
        # Write the values to the file as rows
        for i in range(num_rows):
            row = [i+1] + [stats[key][i] for key in keys]
            writer.writerow(row)

args = parser.parse_args()

if args.mode == 0.0:
    
    numrun = 6
    startseed = 54

# Initialize DQN
# Training mode 1 (3 parameters: cbr, pdr, cf)
if args.mode == 1.0:
    dqn = DQN(3, 2)
    prev_state = [0.0, 0.0, 0.0]
    numrun = 40
    startseed = 4

# Training mode 2, 6 or 10(1 parameter: pdr or cbr)
if args.mode == 2.0 or args.mode == 6.0 or args.mode == 10.0:
    dqn = DQN(1, 2)
    prev_state = [0.0]
    numrun = 24
    startseed = 4
	
# Training mode 8 (2 parameters: cbr, pdr)
if args.mode == 8.0 or args.mode == 12.0:
    dqn = DQN(2, 2)
    prev_state = [0.0, 0.0]
    numrun = 24
    startseed = 4

# Evaluation mode (3 parameters: cbr, pdr, cf)
if args.mode == 3.0:
    dqn = DQN(3, 2)
    prev_state = [0.0, 0.0, 0.0]
    # Load the trained model
    dqn.eval_net.load_state_dict(torch.load('trained_model_1.pth'))

    # Set the network in evaluation mode
    dqn.eval_net.eval()
    
    numrun = 10
    startseed = 54
	
# Evaluation mode 9 (2 parameters: cbr, pdr)
if args.mode == 9.0 or args.mode == 13.0:
    dqn = DQN(2, 2)
    prev_state = [0.0, 0.0]
    # Load the trained model
    dqn.eval_net.load_state_dict(torch.load('trained_model_4.7_cumR.pth'))

    # Set the network in evaluation mode
    dqn.eval_net.eval()
    
    numrun = 6
    startseed = 54

# Evaluation mode 4, 7 or 11 (1 parameter :pdr or cbr)
if args.mode == 4.0 or args.mode == 7.0 or args.mode == 11.0:
    dqn = DQN(1, 2)
    prev_state = [0.0]
    # Load the trained model
    dqn.eval_net.load_state_dict(torch.load('trained_model_5.4_cumR.pth'))

    # Set the network in evaluation mode
    dqn.eval_net.eval()
    
    numrun = 6
    startseed = 54
    
if args.mode == 5.0:
    
    numrun = 10
    startseed = 54

ns3Settings = { 'numSeeds': 4, 'mode': args.mode}
mempool_key = 1234                                          # memory pool key, arbitrary integer large than 1000
#mem_size = 2048 
mem_size = 1024                                           # memory pool size in bytes
memblock_key = 2333                                         # memory block key, need to keep the same in the ns-3 script
exp = Experiment(mempool_key, mem_size, 'wifi-simple-adhoc-grid-b-clean', '../')      # Set up the ns-3 environment

# Initialize prev_actions as an empty list
prev_actions = []

try:
    for i in range(numrun):
        exp.reset()                                             # Reset the environment
        rl = Ns3AIRL(memblock_key, Env, Act)			  # Link the shared memory block with ns-3 script
        
        ns3Settings['numSeeds'] = startseed + i  
            
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
                    reward = compute_reward_1(prev_state, next_state, action, prev_actions)
                    
                    # Update prev_actions
                    prev_actions.append(action)
                    if len(prev_actions) > 2:
                        prev_actions.pop(0)
                    
                    dqn.store_transition(state, action, reward, next_state)
                    if dqn.memory_counter > dqn.memory_capacity:
                        dqn.learn()
                    prev_state = next_state
					
                if args.mode == 2.0:
                
                    state = [pdr]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [pdr]
                    #reward = compute_reward(prev_state, next_state, action)
                    #reward = compute_reward(cbr, pdr, cf, action)
                    
                    # Compute reward with previous actions
                    reward = compute_reward_2(prev_state, next_state, action, prev_actions)
					
                    # Update prev_actions
                    prev_actions.append(action)
                    if len(prev_actions) > 2:
                        prev_actions.pop(0)
                    
                    dqn.store_transition(state, action, reward, next_state)
                    if dqn.memory_counter > dqn.memory_capacity:
                        dqn.learn()
                    prev_state = next_state
                
                if args.mode == 3.0:
                    state = [cbr, pdr, cf]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr, pdr, cf]
                    reward = compute_reward_1(prev_state, next_state, action, prev_actions)
                    prev_state = next_state
					
                if args.mode == 4.0:
                    state = [ pdr ]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [pdr]
                    reward = compute_reward_2(prev_state, next_state, action, prev_actions)
                    prev_state = next_state
                                  
                if args.mode == 5.0:
                    # Random
                    data.act.pred = random.choice([0.0, 1.0])
                    
                if args.mode == 6.0:
                
                    state = [pdr]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [pdr]
                    #reward = compute_reward(prev_state, next_state, action)
                    #reward = compute_reward(cbr, pdr, cf, action)
                    
                    # Compute reward with previous actions
                    reward = compute_reward_4(prev_state, next_state, action, prev_actions)
					
                    # Update prev_actions
                    prev_actions.append(action)
                    if len(prev_actions) > 2:
                        prev_actions.pop(0)
                    
                    dqn.store_transition(state, action, reward, next_state)
                    if dqn.memory_counter > dqn.memory_capacity:
                        dqn.learn()
                    prev_state = next_state
                    
                if args.mode == 7.0:
                    state = [ pdr ]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [pdr]
                    reward = compute_reward_4(prev_state, next_state, action, prev_actions)
                    prev_state = next_state

                if args.mode == 8.0:
                
                    state = [cbr, pdr]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr, pdr]
                    
                    # Compute reward 
                    reward = compute_reward_9(cbr, pdr, action)
					
                    # Update prev_actions
                    prev_actions.append(action)
                    if len(prev_actions) > 2:
                        prev_actions.pop(0)
                    
                    dqn.store_transition(state, action, reward, next_state)
                    if dqn.memory_counter > dqn.memory_capacity:
                        dqn.learn()
                    prev_state = next_state
                    
                if args.mode == 9.0:
                    state = [ cbr, pdr ]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr, pdr]
                    reward = compute_reward_9(cbr, pdr, action)
                    prev_state = next_state	

                if args.mode == 10.0:
                
                    state = [cbr]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr]
                    
                    # Compute reward 
                    reward = compute_reward_6(cbr, action)
					
                    # Update prev_actions
                    prev_actions.append(action)
                    if len(prev_actions) > 2:
                        prev_actions.pop(0)
                    
                    dqn.store_transition(state, action, reward, next_state)
                    if dqn.memory_counter > dqn.memory_capacity:
                        dqn.learn()
                    prev_state = next_state
                    
                if args.mode == 11.0:
                    state = [ cbr ]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr]
                    reward = compute_reward_6(cbr, action)
                    prev_state = next_state	

                if args.mode == 12.0:
                    # here app represents nReps
                    state = [app, cbr]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [app, cbr]
                    
                    # Compute reward 
                    reward = compute_reward_8(app, cbr, action)
					
                    # Update prev_actions
                    prev_actions.append(action)
                    if len(prev_actions) > 2:
                        prev_actions.pop(0)
                    
                    dqn.store_transition(state, action, reward, next_state)
                    if dqn.memory_counter > dqn.memory_capacity:
                        dqn.learn()
                    prev_state = next_state
                    
                if args.mode == 13.0:
                    # here app represents nReps
                    state = [ app, cbr ]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [app, cbr]
                    reward = compute_reward_8(app, cbr, action)
                    prev_state = next_state						
                    
        pro.wait()                                              # Wait the ns-3 to stop
    if args.mode == 1.0:
        #plot_stats(dqn.stats, 'stats_plot_1.png')
        plot_stats(dqn.stats, 'stats_plot_1_cumR.png')
        torch.save(dqn.eval_net.state_dict(), 'trained_model_1.pth')
    if args.mode == 2.0:
        plot_stats(dqn.stats, 'stats_plot_2_cumR.png')
        torch.save(dqn.eval_net.state_dict(), 'trained_model_2.pth')
    if args.mode == 6.0:
        plot_stats(dqn.stats, 'stats_plot_3_cumR.png')
        torch.save(dqn.eval_net.state_dict(), 'trained_model_3_cumR.pth')
    if args.mode == 8.0:
        plot_stats(dqn.stats, 'stats_plot_4.7_cumR.png')
        save_stats_to_tsv(dqn.stats, 'stats_file_4.7_cumR.tsv')
        torch.save(dqn.eval_net.state_dict(), 'trained_model_4.7_cumR.pth')
    if args.mode == 10.0:
        plot_stats(dqn.stats, 'stats_plot_5.4_cumR.png')
        save_stats_to_tsv(dqn.stats, 'stats_file_5.4_cumR.tsv')
        torch.save(dqn.eval_net.state_dict(), 'trained_model_5.4_cumR.pth')
    if args.mode == 12.0:
        plot_stats(dqn.stats, 'stats_plot_6.2_cumR.png')
        save_stats_to_tsv(dqn.stats, 'stats_file_6.2_cumR.tsv')
        torch.save(dqn.eval_net.state_dict(), 'trained_model_6.2_cumR.pth')

except Exception as e:
    print('Something wrong')
    print(e)
finally:
    del exp