from __future__ import print_function
from cloudevents.sdk.event import v1
from dapr.ext.grpc import App
from torch.autograd import Variable
from IPython.display import display
from torchviz import make_dot

import json
# Visualize our data
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

fig = plt.figure()   # Initializes current figure

class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.layer = torch.nn.Linear(1, 1)

   def forward(self, x):
       x = self.layer(x)      
       return x

net = Net()
print(net, flush=True)

app = App()

@app.subscribe(pubsub_name='pubsub', topic='DATA')
def mytopic(event: v1.Event) -> None:
    X = json.loads(event.Data()).get('X')
    Y = json.loads(event.Data()).get('Y')

    # convert numpy array to tensor in shape of input size
    x = torch.from_numpy(np.asarray(X).reshape(-1,1)).float()
    y = torch.from_numpy(np.asarray(Y).reshape(-1,1)).float()

    # Define Optimizer and Loss Function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    inputs = Variable(x)
    outputs = Variable(y)

    for i in range(25):
        prediction = net(inputs)
        loss = loss_func(prediction, outputs) 
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()       

        if i % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
            plt.pause(0.1)

    # display(fig)
    # make_dot(net)
    for param in net.parameters():
        print(param)

app.run(50051)