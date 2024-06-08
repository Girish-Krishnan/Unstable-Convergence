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
batch_size = 32  # Minibatch size
learning_rates = [2 / 50, 2 / 100, 2 / 150]
num_epochs = 300

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                             transform=transform, download=True)
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

# Function to calculate RHS of (5.1)
def calculate_rhs(model, images, labels, criterion, learning_rate):
    model.eval()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([grad.view(-1) for grad in grads])
    
    hessian_vector_product = torch.autograd.grad(grads_flat.norm(), model.parameters(), retain_graph=True)
    hessian_norm = torch.norm(torch.cat([hvp.view(-1) for hvp in hessian_vector_product])).item()
    
    lhs = -1 + (learning_rate / 2) * hessian_norm
    
    model.train()
    return lhs

# Training Function
def train_model(model, train_loader, criterion, learning_rate, num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_history = []
    rp_history = []
    rhs_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_rp = 0
        epoch_rhs = 0
        num_batches = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            rp = calculate_relative_progress(model, images, labels, criterion, learning_rate)
            rhs = calculate_rhs(model, images, labels, criterion, learning_rate)
            epoch_rp += rp
            epoch_rhs += rhs
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        # Average loss, RP, and RHS for the epoch
        epoch_loss /= num_batches
        epoch_rp /= num_batches
        epoch_rhs /= num_batches

        loss_history.append(epoch_loss)
        rp_history.append(epoch_rp)
        rhs_history.append(epoch_rhs)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Expected RP: {epoch_rp:.4f}, RHS: {epoch_rhs:.4f}')

    return loss_history, rp_history, rhs_history

criterion = nn.CrossEntropyLoss()

# Train the Tanh Network with different learning rates
results = {}
for lr in learning_rates:
    print(f"Training Tanh Network with learning rate: {lr}")
    tanh_model = TanhNet2().to(device)
    loss_history, rp_history, rhs_history = train_model(tanh_model, train_loader, criterion, lr, num_epochs)
    results[lr] = (loss_history, rp_history, rhs_history)

# Plot the Results
epochs = list(range(num_epochs))

plt.figure(figsize=(18, 5))

for i, lr in enumerate(learning_rates):
    plt.subplot(1, 3, i + 1)
    plt.plot(epochs, results[lr][1], label='E[RP(θ^t)]', linestyle='dotted', color='red')
    plt.plot(epochs, results[lr][2], label='RHS', linestyle='solid', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'η = {lr}')
    plt.legend()

plt.tight_layout()
plt.savefig('exp_9.png')
plt.show()
