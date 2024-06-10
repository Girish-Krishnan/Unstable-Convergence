import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser
parser = argparse.ArgumentParser(description='Train a neural network with different optimizers and learning rates.')
parser.add_argument('--optimizer', type=str, choices=['Adam', 'RMSProp', 'SGD'], required=True, help='Optimizer to use')
parser.add_argument('--learning_rates', nargs='+', type=float, required=True, help='List of learning rates to use')
parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
parser.add_argument('--dataset_size', type=int, default=500, help='Number of data points in the dataset')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
args = parser.parse_args()

# Hyperparameters
batch_size = args.batch_size
learning_rates = args.learning_rates
num_epochs = args.num_epochs

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             transform=transform, download=True)
train_dataset.data = train_dataset.data[:args.dataset_size]
train_dataset.targets = train_dataset.targets[:args.dataset_size]
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)

# Neural Network with Tanh Activations (2 Hidden Layers)
class TanhNet2(nn.Module):
    def __init__(self):
        super(TanhNet2, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to calculate relative progress
def calculate_relative_progress(model, images, labels, criterion, learning_rate):
    model.eval()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([grad.view(-1) for grad in grads])
    
    new_model = TanhNet2().to(device)  # Create a new instance of the model class
    new_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param, grad in zip(new_model.parameters(), grads):
            param -= learning_rate * grad
    
    new_outputs = new_model(images)
    new_loss = criterion(new_outputs, labels)
    
    rp = (new_loss.item() - loss.item()) / (learning_rate * grads_flat.norm().item()**2)
    
    model.train()
    return rp

# Function to calculate directional smoothness
def calculate_directional_smoothness(model, images, labels, criterion, learning_rate):
    model.eval()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([grad.view(-1) for grad in grads])
    
    new_model = TanhNet2().to(device)  # Create a new instance of the model class
    new_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param, grad in zip(new_model.parameters(), grads):
            param -= learning_rate * grad
    
    new_outputs = new_model(images)
    new_loss = criterion(new_outputs, labels)
    new_grads = torch.autograd.grad(new_loss, new_model.parameters(), create_graph=True)
    new_grads_flat = torch.cat([grad.view(-1) for grad in new_grads])
    
    ds = torch.dot(grads_flat, grads_flat - new_grads_flat).item() / (learning_rate * grads_flat.norm().item()**2)
    
    model.train()
    return ds

# Training function
def train_model(model, train_loader, criterion, optimizer, learning_rate, num_epochs):
    loss_history = []
    rp_history = []
    ds_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_rp = 0
        epoch_ds = 0
        num_batches = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate RP and DS
            rp = calculate_relative_progress(model, images, labels, criterion, learning_rate)
            ds = calculate_directional_smoothness(model, images, labels, criterion, learning_rate)

            epoch_loss += loss.item()
            epoch_rp += rp
            epoch_ds += ds
            num_batches += 1
        
        # Average loss, RP, and DS for the epoch
        epoch_loss /= num_batches
        epoch_rp /= num_batches
        epoch_ds /= num_batches

        loss_history.append(epoch_loss)
        rp_history.append(epoch_rp)
        ds_history.append(epoch_ds)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, RP: {epoch_rp:.4f}, DS: {epoch_ds:.4f}')

    return loss_history, rp_history, ds_history

# Run experiments with specified optimizer
optimizer_classes = {
    'Adam': optim.Adam,
    'RMSProp': optim.RMSprop,
    'SGD': optim.SGD
}

results = {}

optimizer_class = optimizer_classes[args.optimizer]
for lr in learning_rates:
    print(f"Training with {args.optimizer} and learning rate: {lr}")
    model = TanhNet2().to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss_history, rp_history, ds_history = train_model(model, train_loader, criterion, optimizer, lr, num_epochs)
    results[lr] = (loss_history, rp_history, ds_history)

# Plot the results
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes = axes.ravel()

# Add a horizontal line at 0 in RP plot
axes[1].axhline(y=0, color='r', linestyle='--', label='0')

for lr, (loss_history, rp_history, ds_history) in results.items():
    axes[0].plot(range(num_epochs), loss_history, label=f'LR={lr}')
    axes[0].set_title(f'{args.optimizer} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(range(num_epochs), rp_history, label=f'LR={lr}')
    axes[1].set_title(f'{args.optimizer} - RP')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RP')
    axes[1].legend()

    axes[2].plot(range(num_epochs), ds_history, label=f'LR={lr}')
    axes[2].axhline(y=2 / lr, color='b', linestyle='--', label=f'2 / η = {2 / lr:.2f}')
    axes[2].set_title(f'{args.optimizer} - DS')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('DS')
    axes[2].legend()

plt.tight_layout()
plt.savefig('extension.png')
plt.show()
