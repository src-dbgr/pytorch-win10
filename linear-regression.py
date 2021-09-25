import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data

X_numpy, y_numpy = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32)).cuda()
y = torch.from_numpy(y_numpy.astype(np.float32)).cuda()

y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)
model.cuda()

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
criterion.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop

num_epochs = 10000
for epoch in range(num_epochs):
    # forward pas and loss
    y_predicted = model(X).cuda()
    loss = criterion(y_predicted, y).cuda()

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 1000 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item(): .4f}')

# plot
mod = model(X).cpu()
predicted = mod.detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
