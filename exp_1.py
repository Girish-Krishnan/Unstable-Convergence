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
learning_rate = 2 / 30
num_epochs = 10

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)

# Neural Network Architectures
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

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

class ReLUNet(nn.Module):
    def __init__(self):
        super(ReLUNet, self).__init__()
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
def train_model(model, train_loader, criterion, learning_rate, num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_history = []
    sharpness_history = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            sharpness = calculate_sharpness(model, images, labels, criterion)
            sharpness_history.append(sharpness)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Sharpness: {sharpness:.4f}')

    return loss_history, sharpness_history

criterion = nn.CrossEntropyLoss()

# Train Linear Network
linear_model = LinearNet().to(device)
linear_loss_history, linear_sharpness_history = train_model(linear_model, train_loader, criterion, learning_rate, num_epochs)

# Train Tanh Network
tanh_model = TanhNet().to(device)
tanh_loss_history, tanh_sharpness_history = train_model(tanh_model, train_loader, criterion, learning_rate, num_epochs)

# Train ReLU Network
relu_model = ReLUNet().to(device)
relu_loss_history, relu_sharpness_history = train_model(relu_model, train_loader, criterion, learning_rate, num_epochs)

# Plot the Results
iterations = list(range(len(linear_loss_history)))

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(iterations, linear_loss_history, label='Linear Network', linestyle='dotted', color='red')
plt.plot(iterations, tanh_loss_history, label='Tanh Network', linestyle='solid', color='blue')
plt.plot(iterations, relu_loss_history, label='ReLU Network', linestyle='solid', color='green')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

# Plot Sharpness
plt.subplot(1, 2, 2)
plt.plot(iterations, linear_sharpness_history, label='Linear Network', linestyle='dotted', color='red')
plt.plot(iterations, tanh_sharpness_history, label='Tanh Network', linestyle='solid', color='blue')
plt.plot(iterations, relu_sharpness_history, label='ReLU Network', linestyle='solid', color='green')
plt.xlabel('Iteration')
plt.ylabel('Sharpness')
plt.title('Sharpness')
plt.legend()
plt.tight_layout()
plt.savefig('exp_1.png')
plt.show()
