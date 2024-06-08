import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 5
batch_size = 100
learning_rate = 0.1

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                             transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, shuffle=True)

# Simple Neural Network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Deeper Neural Network
class DeeperNet(nn.Module):
    def __init__(self):
        super(DeeperNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Smoothness Aware Optimizer
class SmoothnessAwareOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9):
        defaults = dict(lr=lr, beta=beta)
        super(SmoothnessAwareOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                state['step'] += 1

                exp_avg.mul_(beta).add_(1 - beta, grad)

                # Adjust learning rate based on directional smoothness
                ds = torch.dot(grad.view(-1), exp_avg.view(-1)).item() / (torch.norm(grad.view(-1)).item()**2)
                adjusted_lr = lr / (1 + ds)

                p.data.add_(-adjusted_lr, exp_avg)

        return loss

def compute_relative_progress(model, images, labels, criterion, learning_rate):
    model.train()  # Ensure the model is in training mode
    # Current loss
    outputs = model(images)
    current_loss = criterion(outputs, labels).item()

    # Simulate one step forward
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    # Step forward
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    # New loss
    outputs = model(images)
    new_loss = criterion(outputs, labels).item()

    # Step back
    with torch.no_grad():
        for param in model.parameters():
            param += learning_rate * param.grad

    rp = (new_loss - current_loss) / (learning_rate * sum(torch.norm(param.grad).item()**2 for param in model.parameters()))
    return rp, current_loss, new_loss

def compute_directional_smoothness(model, images, labels, criterion, epsilon=1e-3):
    model.train()  # Ensure the model is in training mode

    # Compute original gradients
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    original_grads = [param.grad.clone() for param in model.parameters()]

    # Apply perturbation and compute new gradients
    with torch.no_grad():
        for param in model.parameters():
            param += epsilon

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    new_grads = [param.grad.clone() for param in model.parameters()]

    # Revert perturbation
    with torch.no_grad():
        for param in model.parameters():
            param -= epsilon

    ds = []
    for original_grad, new_grad in zip(original_grads, new_grads):
        grad_diff = new_grad - original_grad
        ds.append(torch.dot(grad_diff.view(-1), original_grad.view(-1)).item() / (torch.norm(original_grad.view(-1)).item()**2))

    return sum(ds) / len(ds)

# Training function with measurements
def train_and_measure(model, train_loader, criterion, optimizer, learning_rate, num_epochs):
    loss_history = []
    rp_history = []
    ds_history = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Measure RP and DS
            rp, current_loss, new_loss = compute_relative_progress(model, images, labels, criterion, learning_rate)
            ds = compute_directional_smoothness(model, images, labels, criterion)
            rp_history.append(rp)
            ds_history.append(ds)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, RP: {rp:.4f}, DS: {ds:.4f}')
                loss_history.append(loss.item())

    return loss_history, rp_history, ds_history

# Run training and measure
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_history, rp_history, ds_history = train_and_measure(model, train_loader, criterion, optimizer, learning_rate, num_epochs)

# Plotting results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(loss_history)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(rp_history)
plt.xlabel('Step')
plt.ylabel('RP')
plt.title('Relative Progress')

plt.subplot(1, 3, 3)
plt.plot(ds_history)
plt.xlabel('Step')
plt.ylabel('DS')
plt.title('Directional Smoothness')

plt.tight_layout()
plt.show()

# Training with noise
def train_with_noise(model, train_loader, criterion, optimizer, learning_rate, num_epochs):
    loss_history = []
    rp_history = []
    ds_history = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Add noise to the images
            noisy_images = images + torch.randn_like(images) * 0.01

            # Measure RP and DS
            rp, current_loss, new_loss = compute_relative_progress(model, noisy_images, labels, criterion, learning_rate)
            ds = compute_directional_smoothness(model, noisy_images, labels, criterion)
            rp_history.append(rp)
            ds_history.append(ds)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, RP: {rp:.4f}, DS: {ds:.4f}')
                loss_history.append(loss.item())

    return loss_history, rp_history, ds_history

# Choose the model architecture
model = DeeperNet().to(device)  # Use DeeperNet instead of SimpleNet
criterion = nn.CrossEntropyLoss()
optimizer = SmoothnessAwareOptimizer(model.parameters(), lr=learning_rate)  # Use the custom optimizer

# Train with noise
loss_history, rp_history, ds_history = train_with_noise(model, train_loader, criterion, optimizer, learning_rate, num_epochs)

# Plotting results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(loss_history)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(rp_history)
plt.xlabel('Step')
plt.ylabel('RP')
plt.title('Relative Progress')

plt.subplot(1, 3, 3)
plt.plot(ds_history)
plt.xlabel('Step')
plt.ylabel('DS')
plt.title('Directional Smoothness')

plt.tight_layout()
plt.show()
