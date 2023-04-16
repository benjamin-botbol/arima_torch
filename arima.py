import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('your_data_file.csv', index_col=0)
values = data.values.astype('float32')

# Define ARIMA model
class ARIMA(nn.Module):
    def __init__(self, p, d, q):
        super(ARIMA, self).__init__()
        self.ar = nn.Linear(p, 1, bias=False)
        self.ma = nn.Linear(q, 1, bias=False)
        self.d = d

    def forward(self, x):
        ar_term = self.ar(x[:, :-self.d])
        ma_term = self.ma(x[:, -self.d:])
        return ar_term + ma_term

# Define hyperparameters
p = 0
d = 1
q = 1
lr = 0.001
epochs = 1000

# Create ARIMA model
model = ARIMA(p, d, q)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train model
for epoch in range(epochs):
    # Reset gradient
    optimizer.zero_grad()

    # Forward pass
    inputs = torch.tensor(values[:-1])
    targets = torch.tensor(np.diff(values, axis=0))
    outputs = model(inputs)

    # Compute loss and backpropagation
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

# Predict future values
future_inputs = torch.tensor(values[-1:])
for i in range(12):
    future_outputs = model(future_inputs)
    future_inputs = torch.cat((future_inputs, future_outputs), dim=1)

# Plot results
plt.plot(values, label='Original')
plt.plot(np.concatenate((values[:-1], values[-1:] + future_outputs.detach().numpy())), label='Predicted')
plt.legend()
plt.show()
