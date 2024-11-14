"""

This module contain all the functionality to define and setup the agent:
- number of states
- number of actions
- batch size, memory size
- number of step for the target network update
- learning rate
- gamma parameter for the discount of the q-values
- tau parameter for the soft copy of the target network

Classes
--------------------
In that file the following classes are defined:

1. DDQN_Agent
   - create and define the agent parameters using the DDQN algorithm, the gpu, and the method to take actions and to learn

"""

import torch
import torch.nn.functional as F

import numpy as np

from RL.model import DDQN_Graph, ReplayMemory


class DDQN_Agent(): 
    """docstring for ddqn_agent"""
    def __init__(self, n_states, n_actions, batch_size, hidden_size, memory_size, 
                 update_step, learning_rate, gamma, tau):
        super(DDQN_Agent, self).__init__()
        # state space dimension
        self.n_states = n_states
        # action space dimension
        self.n_actions = n_actions
        # configuration
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.update_step = update_step
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau
        # check cpu or gpu
        self.setup_gpu()
        # initialize model graph
        self.setup_model()
        # initialize optimizer
        self.setup_opt()
        # enable Replay Memory
        self.memory = ReplayMemory(memory_size)
        # others
        self.prepare_train()
    
    def setup_gpu(self): 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device utilizzato: ", self.device, "\n")
    
    def setup_model(self):
        self.policy_model = DDQN_Graph(
            self.n_states, 
            self.n_actions, 
            self.hidden_size).to(self.device)
        self.target_model = DDQN_Graph(
            self.n_states, 
            self.n_actions, 
            self.hidden_size).to(self.device)
    
    def setup_opt(self):
        self.opt = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
    
    def prepare_train(self):
        self.steps = 0
    
    def act(self, state, epsilon):
        # take an action for a time step
        # state: 1, state_size
        #print("\nLo stato prima del reshape e':", state, "con dtype = ", np.dtype(state))
        state = torch.tensor(state).reshape(1, -1).to(self.device)
        #print("\nLo stato dopo il reshape e':", state)
        # inference by policy model
        self.policy_model.eval()
        with torch.no_grad(): 
            # action_vs: 1, action_size
            action_vs = self.policy_model(state)
        self.policy_model.train()
        # return action: 1
        # epsilon greedy search
        if np.random.random() > epsilon:
            return np.argmax(action_vs.cpu().detach().numpy())
        else:
            return np.random.randint(self.n_actions)
    
    def step(self, cur_state, action, reward, next_state, done):
        # add one observation to memory
        self.memory.push(cur_state, action, reward, next_state, done)
        # update model for every certain steps
        self.steps = (self.steps + 1) % self.update_step
        if self.steps == 0 and self.memory.size() >= self.batch_size:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            self.learn(states, actions, rewards, next_states, dones)
        else:
            pass
    
    def learn(self, states, actions, rewards, next_states, dones, soft_copy=True):

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # states: batch_size, state_size
        # actions: batch_size, 1
        # rewards: batch_size, 1
        # next_states: batch_size, state_size
        # dones: batch_size, 1
        # target side
        _, next_idx = self.policy_model(next_states).detach().max(1)
        # action values: batch_size, action_size
        target_next_action_vs = self.target_model(next_states).detach().gather(1, next_idx.unsqueeze(1))
        # Q values: batch_size, 1
        # Q = reward + (gamma * Q[next state][next action]) for not done
        target_q_vs = rewards + (self.gamma * target_next_action_vs * (1 - dones))
        # policy side
        # Q values: batch_size, 1
        policy_q_vs = self.policy_model(states).gather(1, actions)
        # compute MSE loss
        loss = F.mse_loss(policy_q_vs, target_q_vs)
        # update policy network
        self.opt.zero_grad()
        loss.backward()
        # gradient clamping
        for p in self.policy_model.parameters(): 
            p.grad.data.clamp_(-1, 1)
        self.opt.step()
        if soft_copy:
            # update target network via soft copy with ratio tau
            # θ_target = τ*θ_local + (1 - τ)*θ_target
            for tp, lp in zip(self.target_model.parameters(), self.policy_model.parameters()):
                tp.data.copy_(self.tau*lp.data + (1.0-self.tau)*tp.data)
        else:
            # update target network via hard copy
            self.target_model.load_state_dict(self.policy_model.state_dict())