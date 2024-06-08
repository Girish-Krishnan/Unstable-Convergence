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
learning_rates = [2 / 30, 2 / 60, 2 / 90]
num_epochs = 20  # We will stop when accuracy reaches 95%
taus = np.arange(0.01, 1.01, 0.01)  # Tau values from 0.01 to 1

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                             transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)

# Neural Network with ReLU Activations (2 Hidden Layers)
class ReLUNet2(nn.Module):
    def __init__(self):
        super(ReLUNet2, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Neural Network with ReLU Activations (4 Hidden Layers)
class ReLUNet4(nn.Module):
    def __init__(self):
        super(ReLUNet4, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Function to calculate relative progress
def calculate_relative_progress(model, images, labels, criterion, learning_rate):
    model.eval()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([grad.view(-1) for grad in grads])
    
    new_model = type(model)().to(device)  # Create a new instance of the model class
    new_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param, grad in zip(new_model.parameters(), grads):
            param -= learning_rate * grad
    
    new_outputs = new_model(images)
    new_loss = criterion(new_outputs, labels)
    
    rp = (new_loss.item() - loss.item()) / (learning_rate * grads_flat.norm().item()**2)
    
    model.train()
    return rp

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
    loss_history = []
    rp_history = []
    ds_tau_history = []
    accuracy = 0

    while accuracy < target_accuracy:
        for iteration, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            rp = calculate_relative_progress(model, images, labels, criterion, learning_rate)
            rp_history.append(rp)
            
            if iteration % 5 == 0:
                ds_tau_values = [measure_directional_smoothness_tau(model, images, labels, criterion, learning_rate, tau) for tau in taus]
                ds_tau_history.append(ds_tau_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = (correct / total) * 100
            
            if accuracy >= target_accuracy:
                break

    return loss_history, rp_history, ds_tau_history

criterion = nn.CrossEntropyLoss()

# Train the ReLU Network with 2 Hidden Layers
results_relu2 = {}
for lr in learning_rates:
    print(f"Training ReLU Network (2 Hidden Layers) with learning rate: {lr}")
    relu_model = ReLUNet2().to(device)
    loss_history, rp_history, ds_tau_history = train_model(relu_model, train_loader, criterion, lr, taus)
    results_relu2[lr] = (loss_history, rp_history, ds_tau_history)

# Train the ReLU Network with 4 Hidden Layers
results_relu4 = {}
for lr in learning_rates:
    print(f"Training ReLU Network (4 Hidden Layers) with learning rate: {lr}")
    relu_model = ReLUNet4().to(device)
    loss_history, rp_history, ds_tau_history = train_model(relu_model, train_loader, criterion, lr, taus)
    results_relu4[lr] = (loss_history, rp_history, ds_tau_history)

# Convert to NumPy array for easier manipulation
for lr in learning_rates:
    results_relu2[lr] = (np.array(results_relu2[lr][0]), np.array(results_relu2[lr][1]), np.array(results_relu2[lr][2]))
    results_relu4[lr] = (np.array(results_relu4[lr][0]), np.array(results_relu4[lr][1]), np.array(results_relu4[lr][2]))

# Compute mean and standard deviation for directional smoothness
ds_tau_mean_relu2 = {lr: np.mean(results_relu2[lr][2], axis=1) for lr in learning_rates}
ds_tau_std_relu2 = {lr: np.std(results_relu2[lr][2], axis=1) for lr in learning_rates}
ds_tau_2std_relu2 = {lr: 2 * ds_tau_std_relu2[lr] for lr in learning_rates}

ds_tau_mean_relu4 = {lr: np.mean(results_relu4[lr][2], axis=1) for lr in learning_rates}
ds_tau_std_relu4 = {lr: np.std(results_relu4[lr][2], axis=1) for lr in learning_rates}
ds_tau_2std_relu4 = {lr: 2 * ds_tau_std_relu4[lr] for lr in learning_rates}

# Plot the Results
iterations_relu2 = {lr: list(range(len(results_relu2[lr][0]))) for lr in learning_rates}
iterations_relu4 = {lr: list(range(len(results_relu4[lr][0]))) for lr in learning_rates}

plt.figure(figsize=(18, 10))

# Plot Loss for ReLU Network (2 Hidden Layers)
plt.subplot(2, 3, 1)
for lr in learning_rates:
    plt.plot(iterations_relu2[lr], results_relu2[lr][0], label=f'η = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss (ReLU Network, 2 Hidden Layers)')
plt.legend()

# Plot Relative Progress for ReLU Network (2 Hidden Layers)
plt.subplot(2, 3, 2)
for lr in learning_rates:
    plt.plot(iterations_relu2[lr], results_relu2[lr][1], label=f'η = {lr}')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Relative Progress')
plt.title('Relative Progress (ReLU Network, 2 Hidden Layers)')
plt.legend()

# Plot Directional Smoothness for ReLU Network (2 Hidden Layers)
plt.subplot(2, 3, 3)
for lr in learning_rates:
    plt.plot(iterations_relu2[lr], ds_tau_mean_relu2[lr], label=f'η = {lr}')
    plt.fill_between(iterations_relu2[lr], ds_tau_mean_relu2[lr] - ds_tau_std_relu2[lr], ds_tau_mean_relu2[lr] + ds_tau_std_relu2[lr], alpha=0.3)
    plt.fill_between(iterations_relu2[lr], ds_tau_mean_relu2[lr] - ds_tau_2std_relu2[lr], ds_tau_mean_relu2[lr] + ds_tau_2std_relu2[lr], alpha=0.1)
plt.axhline(y=2 / min(learning_rates), color='r', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel(r'$L(\theta^t; \eta \tau \nabla f(\theta^t))$')
plt.title('Directional Smoothness (ReLU Network, 2 Hidden Layers)')
plt.legend()

# Plot Loss for ReLU Network (4 Hidden Layers)
plt.subplot(2, 3, 4)
for lr in learning_rates:
    plt.plot(iterations_relu4[lr], results_relu4[lr][0], label=f'η = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss (ReLU Network, 4 Hidden Layers)')
plt.legend()

# Plot Relative Progress for ReLU Network (4 Hidden Layers)
plt.subplot(2, 3, 5)
for lr in learning_rates:
    plt.plot(iterations_relu4[lr], results_relu4[lr][1], label=f'η = {lr}')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Relative Progress')
plt.title('Relative Progress (ReLU Network, 4 Hidden Layers)')
plt.legend()

# Plot Directional Smoothness for ReLU Network (4 Hidden Layers)
plt.subplot(2, 3, 6)
for lr in learning_rates:
    plt.plot(iterations_relu4[lr], ds_tau_mean_relu4[lr], label=f'η = {lr}')
    plt.fill_between(iterations_relu4[lr], ds_tau_mean_relu4[lr] - ds_tau_std_relu4[lr], ds_tau_mean_relu4[lr] + ds_tau_std_relu4[lr], alpha=0.3)
    plt.fill_between(iterations_relu4[lr], ds_tau_mean_relu4[lr] - ds_tau_2std_relu4[lr], ds_tau_mean_relu4[lr] + ds_tau_2std_relu4[lr], alpha=0.1)
plt.axhline(y=2 / min(learning_rates), color='r', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel(r'$L(\theta^t; \eta \tau \nabla f(\theta^t))$')
plt.title('Directional Smoothness (ReLU Network, 4 Hidden Layers)')
plt.legend()

plt.tight_layout()
plt.savefig('exp_6.png')
plt.show()
