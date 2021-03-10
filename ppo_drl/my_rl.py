import torch 
from torch import nn
import torch.nn.functional as F 
import numpy as np 

class Environment():
    def __init__(self, variables, set_points, system , random = True):
        self.initial_set_points = set_points
        self.initial_variables  = variables
        self.refresh(self, random)

        self.model = model
    
    def refresh(self, random = True):
        self.set_points = self.initial_set_points
        self.variables  = self.initial_variables

        if random:
            self.variables *= (1 + 0.1*np.random.rand(len(self.variables)))

        self.state = (self.system.measured @ self.variables  - self.set_points)/self.set_points

        
    def update(self, action):
        # Updates the manipulated variables and performs one integration step
        self.variables *= self.system.manipulated @ (action - 0.5)
        self.variables  = self.system.equations(self.variables)


class System():
    def __init__(self, measured, manipulated):
        self.measured    = self.to_tensor(measured)
        self.manipulated = self.to_tensor(manipulated).t()


    def to_tensor(self, vector):
        matrix = []

        for ii in range(len(vector)):
            if vector[ii]:
                row = [1 if jj == ii else 0 for jj in range(len(vector))]
                matrix.append(row)

        return torch.tensor(matrix, dtype=torch.float64)

    
    def equations(self, variables):
        # Generates the next-step variables values, based on the ODE governing equations
        # Variables = [h, Fin, Fout, Fin2, A, alpha]

        for _ in range(10):
            Fout = alpha*h**0.5
            dh = (Fin - Fout)/A*0.1
            h += dh

        variables[0] = h
        variables[2] = Fout

        return variables


class Policy(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, layers_num):
        super(Policy, self).__init__()

        self.layer[0] = nn.Linear(in_dim, hidden_dim)

        for ii in range(1, layers_num - 1):
            self.layer[ii] = nn.Linear(hidden_dim, hidden_dim)

        self.layer[layers_num] = nn.Linear(hidden_dim, out_dim)


    def get_action(self, state):
        # Returns which actions to take, based on the current policy

        # Convert observation to tensor
        if isinstance(state, np.ndarray):
            obs = torch.tensor(state, dtype = torch.float)

        activation = F.sigmoid(self.layer(state))
        for ii in range(1, layers_num):
            activation = F.sigmoid(self.layer2(activation))

        return activation

    
class Agent():
    def __init__(self, policy, environment):
        self.policy      = policy
        self.environment = environment


    def observe(self):
        # Get the current state by observing the environment
        self.state = self.environment.state

    
    def think(self):
        # Ask the policy which action to take
        self.action = self.policy.get_action(self.state)


    def move(self):
        # Change th state by acting on the environment
        self.environment.update(self.action)


class Trainer():
    def __init__(self, policy):
        self.policy    = policy
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=1e-5)


    def improve(self, outcome):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


class Gim():
    def __init__(self, trainer, agent):
        self.trainer     = trainer
        self.policy      = agent.policy
        self.agent       = agent
        self.environment = agent.environment


    def training(self, time, time_per_episode):
        n_episodes = time//time_per_episode

        for ii in range(n_episodes):
            self.environment.refresh()

            for _ in range(time_per_episode):
                self.agent.observe()
                self.agent.think()
                self.agent.move()
            
            self.trainer.improve(outcome)

            if ii + 1 % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(ii+1, n_batches, loss.item()))

    
    def routine(self, time, time_per_batch, time_per_episode):
        n_episodes = time_per_batch//time_per_episode

        for _ in range(n_episodes):
            
                