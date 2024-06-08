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
batch_size = 5000  # Full-batch
learning_rate = 2 / 60
num_epochs = 20  # We will stop when accuracy reaches 95%
taus = np.arange(0.01, 1.01, 0.01)  # Tau values from 0.01 to 1

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)

# Neural Network with Tanh Activations
class TanhNet(nn.Module):
    def __init__(self):
        super(TanhNet, self).__init__()
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

# Function to measure directional smoothness with tau
def measure_directional_smoothness_tau(model, images, labels, criterion, learning_rate, tau):
    model.eval()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([grad.view(-1) for grad in grads])
    
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            param -= learning_rate * tau * grad
    
    new_outputs = model(images)
    new_loss = criterion(new_outputs, labels)
    
    ds_tau = new_loss.item()
    
    # Revert the model parameters
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            param += learning_rate * tau * grad
    
    model.train()
    return ds_tau

# Training Function
def train_model(model, train_loader, criterion, learning_rate, taus, target_accuracy=95):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    ds_tau_history = []
    accuracy = 0

    while accuracy < target_accuracy:
        for iteration, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if iteration % 5 == 0:
                ds_tau_values = [measure_directional_smoothness_tau(model, images, labels, criterion, learning_rate, tau) for tau in taus]
                ds_tau_history.append(ds_tau_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = (correct / total) * 100
            
            if accuracy >= target_accuracy:
                break

    return ds_tau_history

criterion = nn.CrossEntropyLoss()

# Train the Tanh Network
print(f"Training with learning rate: {learning_rate}")
tanh_model = TanhNet().to(device)
ds_tau_history = train_model(tanh_model, train_loader, criterion, learning_rate, taus)

# Convert to NumPy array for easier manipulation
ds_tau_history = np.array(ds_tau_history)

# Compute mean and standard deviation
ds_tau_mean = np.mean(ds_tau_history, axis=1)
ds_tau_std = np.std(ds_tau_history, axis=1)
ds_tau_2std = 2 * ds_tau_std

# Plot the Results
iterations = list(range(len(ds_tau_mean)))

plt.figure(figsize=(10, 5))

plt.plot(iterations, ds_tau_mean, label=r'Mean $L(\theta^t; \eta \tau \nabla f(\theta^t))$', color='black')
plt.fill_between(iterations, ds_tau_mean - ds_tau_std, ds_tau_mean + ds_tau_std, color='blue', alpha=0.3, label=r'$\sigma$')
plt.fill_between(iterations, ds_tau_mean - ds_tau_2std, ds_tau_mean + ds_tau_2std, color='blue', alpha=0.1, label=r'$2\sigma$')

plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel(r'$L(\theta^t; \eta \tau \nabla f(\theta^t))$')
plt.title(r'Directional Smoothness with Tau')
plt.legend()

plt.tight_layout()
plt.savefig('exp_5.png')
plt.show()
