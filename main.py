import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# general settings
dtype = torch.float             # data type
device = torch.device('cpu')    # device to run on
x_range = [0, 5]                # x-range of the problem

# Exact solution for the ODE: (Wolfram...)
# y = 5x^2/(1+x^5)
x = torch.linspace(x_range[0], x_range[1], 100).view(-1, 1)
y = 5 * x**2 / (1 + x**5)

# upload measured data
xm = torch.tensor(np.load('x.npy'), dtype=dtype, device=device).view(-1, 1)
ym = torch.tensor(np.load('y.npy'), dtype=dtype, device=device).view(-1, 1)

# points for the physics informed training
phys_n = xm.size(0)
x_phys = torch.linspace(x_range[0], x_range[1], phys_n, dtype=dtype, device=device).view(-1, 1)
x_phys = x_phys[x_phys != 0].view(-1, 1)    # the ODE is not defined in x=0
x_phys.requires_grad = True

# define a fully connected network:
# 3 hidden layers with 32 neurons each
model = nn.Sequential(*[nn.Linear(1, 32), nn.Tanh(),    # input to hidden1
                        nn.Linear(32, 32), nn.Tanh(),   # hidden1 to hidden2
                        nn.Linear(32, 32), nn.Tanh(),   # hidden2 to hidden3
                        nn.Linear(32, 1)])              # hidden3 to output

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# train model
steps = 10000
for t in range(steps):
    optimizer.zero_grad()

    # run the data points through the network and compute MSE loss
    yp = model(xm)
    loss_data = (yp - ym).pow(2).mean()

    # run the physics points through the model and compute physics loss
    y_phys = model(x_phys)
    dy_phys = torch.autograd.grad(y_phys, x_phys, torch.ones_like(y_phys), create_graph=True)[0]
    loss_phys = (dy_phys + x_phys**2 * y_phys**2 - 2 * y_phys / x_phys).pow(2).mean()

    # train step according to combined loss
    loss = loss_data + loss_phys
    loss.backward()
    optimizer.step()

    if (t+1) % 100 == 0:
        print(round(100*(t+1)/steps, 1), '%    loss =', loss.item())


# plot and save a final figure
plt.plot(x, y, color='black')       # exact solution in black
plt.scatter(xm, ym, color='red')    # measurements in red
yp_full = model(x).detach()         # predicted function in blue
plt.plot(x, yp_full, color='blue', linewidth=5, alpha=0.5)
plt.savefig('Final.png')
plt.show()
