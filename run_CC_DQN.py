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
parser.add_argument('--numNodes', type=int,
                    default=16, help='number of nodes nxn; 3<=n<=5')
parser.add_argument('--mode', type=float,
                    default=3.0, help='simulation mode')
parser.add_argument('--topology', type=str,
                    default="SquareGrid", help='topology type')
parser.add_argument('--traffic', type=str,
                    default="Random", help='traffic pattern')
parser.add_argument('--predInterval', type=float,
                    default=1.0, help='interval time (s) to make prediction')

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
		
    def save_model(self, path):
        torch.save({
            'eval_net_state_dict': self.eval_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.stats['MSE Loss'][-1] if self.stats['MSE Loss'] else None,
            'memory': self.memory,
            'memory_counter': self.memory_counter,
            'cumulative_return': self.cumulative_return,
            'stats': self.stats,
            'learn_step': self.learn_step
            }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.eval_net.load_state_dict(checkpoint['eval_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['loss'] is not None:
            self.stats['MSE Loss'].append(checkpoint['loss'])
        self.memory = checkpoint['memory']
        self.memory_counter = checkpoint['memory_counter']
        self.cumulative_return = checkpoint['cumulative_return']
        self.stats = checkpoint['stats']
        self.learn_step = checkpoint['learn_step']

        # You need to set eval_net and target_net to eval mode if you want to use them for inference
        self.eval_net.eval()
        self.target_net.eval()

def compute_reward_1(cbr, action):
    
    if cbr < 0.1:
        total_reward = (1-cbr) * action 

    if cbr >= 0.1:
        total_reward = (1-cbr) * (1 - action) 
    
    return total_reward
    
def compute_reward_2(cbr, pdr, action):
    
    if cbr < 0.1:
        total_reward = ((1-cbr) * action) + (pdr * action)

    if cbr >= 0.1:
        total_reward = (cbr * (1 - action)) + (pdr * (1 - action))
    
    return total_reward
	
def compute_reward_3(pdrGlobal, pdrNode, action):
    
    if pdrNode < pdrGlobal:
        total_reward = (action) * (pdrGlobal + (1 - pdrNode))

    if pdrNode >= pdrGlobal:
        total_reward = (1 - action) * ((1 - pdrGlobal) + pdrNode)
    
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
# Evaluation mode 1 (1 parameter: cbr)
if args.mode == 1.0:
    dqn = DQN(1, 2)
    # Load the trained model
    #dqn.eval_net.load_state_dict(torch.load('trained_model_1.0.pth'))
    dqn.load_model('trained_model_1.0.pth')
    # Set the network in evaluation mode
    #dqn.eval_net.eval()
    numrun = 6
    startseed = 54

# Evaluation mode 2 (2 parameter: cbr and pdr)
if args.mode == 2.0:
    dqn = DQN(2, 2)
    # Load the trained model
    #dqn.eval_net.load_state_dict(torch.load('trained_model_2.0.pth'))
    dqn.load_model('trained_model_2.0.pth')
    # Set the network in evaluation mode
    #dqn.eval_net.eval()
    numrun = 6
    startseed = 54
	
# Evaluation mode 5 (1 parameter: cbr)
if args.mode == 5.0:
    dqn = DQN(1, 2)
    # Load the trained model
    #dqn.eval_net.load_state_dict(torch.load('trained_model_1.1.pth'))
    #dqn.eval_net = torch.jit.load('trained_model_1.2.pt')
    #dqn.eval_net.load_state_dict(torch.load('trained_model_1.3.pth'))
    dqn.load_model('trained_model_1.4.pth')
	
    # Set the network in evaluation mode
    #dqn.eval_net.eval()
    numrun = 6
    startseed = 54

# Evaluation mode 6 (2 parameter: cbr and pdr)
if args.mode == 6.0:
    dqn = DQN(2, 2)
    # Load the trained model
    #dqn.eval_net.load_state_dict(torch.load('trained_model_2.1.pth'))
    #dqn.eval_net = torch.jit.load('trained_model_2.2.pt')
    #dqn.eval_net.load_state_dict(torch.load('trained_model_2.3.pth'))
    dqn.load_model('trained_model_2.4.pth')

    # Set the network in evaluation mode
    dqn.eval_net.eval()
    numrun = 6
    startseed = 54	

# Evaluation mode 7 (2 parameter: pdrGlobal and pdr)
if args.mode == 7.0:
    dqn = DQN(2, 2)
    # Load the trained model
    #dqn.eval_net.load_state_dict(torch.load('trained_model_2.1.pth'))
    #dqn.eval_net = torch.jit.load('trained_model_2.2.pt')
    #dqn.eval_net.load_state_dict(torch.load('trained_model_3.0.pth'))
    dqn.load_model('trained_model_3.0.pth')

    # Set the network in evaluation mode
    dqn.eval_net.eval()
    numrun = 6
    startseed = 54

# Training mode 3 (1 parameter: cbr)
if args.mode == 3.0:
    dqn = DQN(1, 2)
    numrun = 24
    startseed = 4

# Training mode 4 (2 parameters: cbr and pdr)
if args.mode == 4.0 or args.mode == 8:
    dqn = DQN(2, 2)
    numrun = 24
    startseed = 4

ns3Settings = { 'numSeeds': 4, 'mode': args.mode, 'topology': args.topology, 'TrafficPattern': args.traffic, 'predictInterval': args.predInterval, 'numNodes': args.numNodes }
mempool_key = 1234                                          # memory pool key, arbitrary integer large than 1000
#mem_size = 2048 
mem_size = 1024                                           # memory pool size in bytes
memblock_key = 2345                                        # memory block key, need to keep the same in the ns-3 script
exp = Experiment(mempool_key, mem_size, 'wifi-adhoc-b-udp-cc-dqn', '../')      # Set up the ns-3 environment

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
                    
                if args.mode == 1.0 or args.mode == 5.0:
                    state = [ cbr ]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr]
                    reward = compute_reward_1(cbr, action)
					
                if args.mode == 2.0 or args.mode == 6.0:
                    state = [ cbr, pdr ]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr, pdr]
                    reward = compute_reward_2(cbr, pdr, action)
                
                if args.mode == 3.0:
                
                    state = [cbr]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr]
                    
                    # Compute reward 
                    reward = compute_reward_1(cbr, action)
                    
                    dqn.store_transition(state, action, reward, next_state)
                    if dqn.memory_counter > dqn.memory_capacity:
                        dqn.learn()
					
                if args.mode == 4.0:
                
                    state = [cbr, pdr]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [cbr, pdr]
                    
                    # Compute reward 
                    reward = compute_reward_2(cbr, pdr, action)
                    
                    dqn.store_transition(state, action, reward, next_state)
                    if dqn.memory_counter > dqn.memory_capacity:
                        dqn.learn()			

                if args.mode == 8.0:
                    pdrGlobal = app
                    state = [pdrGlobal, pdr]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [pdrGlobal, pdr]
                    
                    # Compute reward 
                    reward = compute_reward_3(pdrGlobal, pdr, action)
                    
                    dqn.store_transition(state, action, reward, next_state)
                    if dqn.memory_counter > dqn.memory_capacity:
                        dqn.learn()		

                if args.mode == 7.0:
                    pdrGlobal = app
                    state = [pdrGlobal, pdr]
                    action = dqn.choose_action(state)
                    data.act.pred = action
                    next_state = [pdrGlobal, pdr]
                    reward = compute_reward_3(pdrGlobal, pdr, action)						
                    
        pro.wait()                                              # Wait the ns-3 to stop

    if args.mode == 3.0:
        plot_stats(dqn.stats, 'stats_plot_1.4.png')
        save_stats_to_tsv(dqn.stats, 'stats_file_1.4.tsv')
        #torch.save(dqn.eval_net.state_dict(), 'trained_model_1.1.pth')
        #model_scripted = torch.jit.script(dqn.eval_net) # Export to TorchScript
        #model_scripted.save('trained_model_1.2.pt') # Save
        #torch.save(dqn.eval_net, 'trained_model_1.3.pth')
        dqn.save_model('trained_model_1.4.pth')

    if args.mode == 4.0:
        plot_stats(dqn.stats, 'stats_plot_2.4.png')
        save_stats_to_tsv(dqn.stats, 'stats_file_2.4.tsv')
        #torch.save(dqn.eval_net.state_dict(), 'trained_model_2.1.pth')
        #model_scripted = torch.jit.script(dqn.eval_net) # Export to TorchScript
        #model_scripted.save('trained_model_2.2.pt') # Save
        #torch.save(dqn.eval_net, 'trained_model_2.3.pth')
        dqn.save_model('trained_model_2.4.pth')

    if args.mode == 8.0:
        plot_stats(dqn.stats, 'stats_plot_3.0.png')
        save_stats_to_tsv(dqn.stats, 'stats_file_3.0.tsv')
        dqn.save_model('trained_model_3.0.pth')

except Exception as e:
    print('Something wrong')
    print(e)
finally:
    del exp