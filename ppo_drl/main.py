from my_rl import *
import torch

# Variables = [h, Fin, Fout, Fin2, A, alpha]

measured    = [1, 0, 0, 0, 0, 0]
manipulated = [0, 0, 0, 1, 0, 0]

variables  = torch.tensor([8, 0, 80/3600, 80/3600, 4, 5e-2])
set_points = torch.tensor([8.])

system = System(measured, manipulated)
environment = Environment(variables, set_points, system)
policy = Policy(len(variables), out_dim, hidden_dim, layers_num)
controller = Agent(policy, environment)
trainer = Trainer(policy)
gim = Gim(trainer, agent)

gim.training(48000, 480)