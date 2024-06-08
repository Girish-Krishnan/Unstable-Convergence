import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 5000  # Full-batch
learning_rates_stable = [2 / 300, 2 / 400, 2 / 500]
learning_rates_unstable = [2 / 30, 2 / 60, 2 / 90]
num_epochs = 20  # We will stop when accuracy reaches 95%

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True,
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

# Function to measure directional smoothness
def measure_directional_smoothness(model, images, labels, criterion, learning_rate):
    model.eval()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([grad.view(-1) for grad in grads])
    
    model.train()
    return loss.item() - learning_rate * grads_flat.norm().item()

# Training Function
def train_model(model, train_loader, criterion, learning_rate, target_accuracy=95):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_history = []
    directional_smoothness_history = []
    accuracy = 0

    while accuracy < target_accuracy:
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            ds = measure_directional_smoothness(model, images, labels, criterion, learning_rate)
            directional_smoothness_history.append(ds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = (correct / total) * 100

            print(f'Accuracy: {accuracy:.2f}%', end='\r')
            
            if accuracy >= target_accuracy:
                break

    return loss_history, directional_smoothness_history

criterion = nn.CrossEntropyLoss()

# Train the Tanh Network with different learning rates
results_stable = {}
results_unstable = {}

for lr in learning_rates_stable:
    print(f"Training with learning rate: {lr}")
    tanh_model = TanhNet().to(device)
    loss_history, directional_smoothness_history = train_model(tanh_model, train_loader, criterion, lr)
    results_stable[lr] = (loss_history, directional_smoothness_history)

for lr in learning_rates_unstable:
    print(f"Training with learning rate: {lr}")
    tanh_model = TanhNet().to(device)
    loss_history, directional_smoothness_history = train_model(tanh_model, train_loader, criterion, lr)
    results_unstable[lr] = (loss_history, directional_smoothness_history)

# Plot the Results
iterations_stable = {lr: list(range(len(results_stable[lr][0]))) for lr in learning_rates_stable}
iterations_unstable = {lr: list(range(len(results_unstable[lr][0]))) for lr in learning_rates_unstable}

plt.figure(figsize=(18, 5))

# Plot Directional Smoothness for Stable Regime
plt.subplot(1, 2, 1)
for lr in learning_rates_stable:
    plt.plot(iterations_stable[lr], results_stable[lr][1], label=f'η = {lr}')
plt.xlabel('Iteration')
plt.ylabel(r'$L(\theta^t; \eta \nabla f(\theta^t))$')
plt.title(r'Directional Smoothness (Stable Regime)')
plt.legend()

# Plot Directional Smoothness for Unstable Regime
plt.subplot(1, 2, 2)
for lr in learning_rates_unstable:
    plt.plot(iterations_unstable[lr], results_unstable[lr][1], label=f'η = {lr}')
plt.axhline(y=2 / min(learning_rates_unstable), color='r', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel(r'$L(\theta^t; \eta \nabla f(\theta^t))$')
plt.title(r'Directional Smoothness (Unstable Regime)')
plt.legend()

plt.tight_layout()
plt.savefig('exp_4.png')
plt.show()
