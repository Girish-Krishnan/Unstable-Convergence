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
learning_rates = [2 / 90, 2 / 60, 2 / 30]
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

# Function to calculate relative progress
def calculate_relative_progress(model, images, labels, criterion, learning_rate):
    model.eval()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([grad.view(-1) for grad in grads])
    
    new_model = TanhNet().to(device)
    new_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param, grad in zip(new_model.parameters(), grads):
            param -= learning_rate * grad
    
    new_outputs = new_model(images)
    new_loss = criterion(new_outputs, labels)
    
    rp = (new_loss.item() - loss.item()) / (learning_rate * grads_flat.norm().item()**2)
    
    model.train()
    return rp

# Function to calculate sharpness approximation
def calculate_sharpness(model, images, labels, criterion):
    model.eval()
    outputs = model(images)
    loss = criterion(outputs, labels)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads = torch.cat([grad.view(-1) for grad in grads])
    hessian_vector_product = torch.autograd.grad(grads.norm(), model.parameters(), retain_graph=True)
    sharpness = torch.norm(torch.cat([hvp.view(-1) for hvp in hessian_vector_product])).item()
    model.train()
    return sharpness

# Training Function
def train_model(model, train_loader, criterion, learning_rate, target_accuracy=95):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_history = []
    rp_history = []
    sharpness_history = []
    accuracy = 0

    while accuracy < target_accuracy:
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            rp = calculate_relative_progress(model, images, labels, criterion, learning_rate)
            sharpness = calculate_sharpness(model, images, labels, criterion)
            rp_history.append(rp)
            sharpness_history.append(sharpness)

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

    return loss_history, rp_history, sharpness_history

criterion = nn.CrossEntropyLoss()

# Train the Tanh Network with different learning rates
results = {}

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    tanh_model = TanhNet().to(device)
    loss_history, rp_history, sharpness_history = train_model(tanh_model, train_loader, criterion, lr)
    results[lr] = (loss_history, rp_history, sharpness_history)

# Plot the Results
iterations = {lr: list(range(len(results[lr][0]))) for lr in learning_rates}

plt.figure(figsize=(18, 5))

# Plot Loss
plt.subplot(1, 3, 1)
for lr in learning_rates:
    plt.plot(iterations[lr], results[lr][0], label=f'η = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

# Plot Relative Progress
plt.subplot(1, 3, 2)
for lr in learning_rates:
    plt.plot(iterations[lr], results[lr][1], label=f'η = {lr}')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Relative Progress')
plt.title('Relative Progress')
plt.legend()

# Plot Sharpness
plt.subplot(1, 3, 3)
for lr in learning_rates:
    plt.plot(iterations[lr], results[lr][2], label=f'η = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Sharpness')
plt.title('Sharpness')
plt.legend()

plt.tight_layout()
plt.savefig('exp_3.png')
plt.show()
