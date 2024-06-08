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
learning_rate = 2 / 100
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

# Function to calculate relative progress
def calculate_relative_progress(model, images, labels, criterion, learning_rate):
    model.eval()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([grad.view(-1) for grad in grads])
    
    new_model = ReLUNet2().to(device)  # Create a new instance of the model class
    new_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param, grad in zip(new_model.parameters(), grads):
            param -= learning_rate * grad
    
    new_outputs = new_model(images)
    new_loss = criterion(new_outputs, labels)
    
    rp = (new_loss.item() - loss.item()) / (learning_rate * grads_flat.norm().item()**2)
    
    model.train()
    return rp

# Training Function
def train_model(model, train_loader, criterion, learning_rate, num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_history = []
    rp_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_rp = 0
        num_batches = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            rp = calculate_relative_progress(model, images, labels, criterion, learning_rate)
            epoch_rp += rp
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        # Average loss and RP for the epoch
        epoch_loss /= num_batches
        epoch_rp /= num_batches

        loss_history.append(epoch_loss)
        rp_history.append(epoch_rp)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Expected RP: {epoch_rp:.4f}')

    return loss_history, rp_history

criterion = nn.CrossEntropyLoss()

# Train the ReLU Network
print(f"Training ReLU Network with learning rate: {learning_rate}")
relu_model = ReLUNet2().to(device)
loss_history, rp_history = train_model(relu_model, train_loader, criterion, learning_rate, num_epochs)

# Plot the Results
epochs = list(range(num_epochs))

plt.figure(figsize=(18, 5))

# Plot Loss
plt.subplot(1, 3, 1)
plt.plot(epochs, loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

# Plot Expected Relative Progress
plt.subplot(1, 3, 2)
plt.plot(epochs, rp_history, label='Expected RP')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Expected RP')
plt.title('Expected Relative Progress')
plt.legend()

# Plot Zoomed-in Expected Relative Progress
plt.subplot(1, 3, 3)
plt.plot(epochs, rp_history, label='Expected RP')
plt.axhline(y=0, color='r', linestyle='--')
plt.ylim([-0.5, 0.5])
plt.xlabel('Epoch')
plt.ylabel('Expected RP')
plt.title('Expected Relative Progress (Zoomed-in)')
plt.legend()

plt.tight_layout()
plt.savefig('exp_7.png')
plt.show()
