import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

# Function to compute relative progress
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

# Function to compute directional smoothness
def compute_directional_smoothness(model, images, labels, criterion, epsilon=1e-3, small_value=1e-10):
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
        norm = torch.norm(original_grad.view(-1)).item() + small_value  # Add small_value to prevent division by zero
        ds.append(torch.dot(grad_diff.view(-1), original_grad.view(-1)).item() / (norm**2))

    return sum(ds) / len(ds)

# Function to verify the relation
def verify_relation(model, images, labels, criterion, learning_rate):
    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    
    rp = compute_relative_progress(model, images, labels, criterion, learning_rate)[0]
    ds = compute_directional_smoothness(model, images, labels, criterion)
    
    return rp, ds

# Plotting results
def plot_results(loss_history, rp_history, ds_history):
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

# Training with the new optimizer
def train_with_smoothness_aware_optimizer(model, train_loader, criterion, learning_rate, num_epochs):
    optimizer = SmoothnessAwareOptimizer(model.parameters(), lr=learning_rate)
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

import time

def compare_optimizers(model_class, train_loader, criterion, learning_rate, num_epochs, loss_threshold):
    # Original optimizer
    model = model_class().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    start_time = time.time()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() <= loss_threshold:
                original_time = time.time() - start_time
                break

    # Smoothness-aware optimizer
    model = model_class().to(device)
    optimizer = SmoothnessAwareOptimizer(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() <= loss_threshold:
                smoothness_aware_time = time.time() - start_time
                break

    return original_time, smoothness_aware_time

# Function to compute gradient and Hessian Lipschitz constants
def compute_lipschitz_constants(model, images, labels, criterion, epsilon=1e-3):
    model.train()

    # Compute gradient Lipschitz constant
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    original_grads = [param.grad.clone() for param in model.parameters()]

    with torch.no_grad():
        for param in model.parameters():
            param += epsilon

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    new_grads = [param.grad.clone() for param in model.parameters()]

    grad_lipschitz = max(torch.norm(new_grad - original_grad).item() / epsilon for original_grad, new_grad in zip(original_grads, new_grads))

    # Compute Hessian Lipschitz constant
    hessian_lipschitz = 0
    for param in model.parameters():
        param.grad.zero_()
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward(create_graph=True)
    original_hessians = [param.grad.clone() for param in model.parameters()]

    with torch.no_grad():
        for param in model.parameters():
            param += epsilon

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward(create_graph=True)
    new_hessians = [param.grad.clone() for param in model.parameters()]

    hessian_lipschitz = max(torch.norm(new_hessian - original_hessian).item() / epsilon for original_hessian, new_hessian in zip(original_hessians, new_hessians))

    return grad_lipschitz, hessian_lipschitz

# Function to train with Adam optimizer
def train_with_adam(model_class, train_loader, criterion, learning_rate, num_epochs):
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    rp_history = []
    ds_history = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            rp, current_loss, new_loss = compute_relative_progress(model, images, labels, criterion, learning_rate)
            ds = compute_directional_smoothness(model, images, labels, criterion)
            rp_history.append(rp)
            ds_history.append(ds)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, RP: {rp:.4f}, DS: {ds:.4f}')
                loss_history.append(loss.item())

    return loss_history, rp_history, ds_history

def train_with_optimizer(model_class, train_loader, criterion, optimizer_class, learning_rate, num_epochs):
    model = model_class().to(device)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    loss_history = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                loss_history.append(loss.item())

    return loss_history

if __name__ == "__main__":
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    loss_history, rp_history, ds_history = train_and_measure(model, train_loader, criterion, optimizer, learning_rate, num_epochs)
    plot_results(loss_history, rp_history, ds_history)

    # Train with the smoothness aware optimizer
    model = SimpleNet().to(device)
    loss_history, rp_history, ds_history = train_with_smoothness_aware_optimizer(model, train_loader, criterion, learning_rate, num_epochs)
    plot_results(loss_history, rp_history, ds_history)

    # Compare optimizers
    loss_threshold = 1.0
    original_time, smoothness_aware_time = compare_optimizers(SimpleNet, train_loader, nn.CrossEntropyLoss(), learning_rate, num_epochs, loss_threshold)
    print(f"Original optimizer time: {original_time:.2f} seconds")
    print(f"Smoothness-aware optimizer time: {smoothness_aware_time:.2f} seconds")

    # Measure Lipschitz constants
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    grad_lipschitz, hessian_lipschitz = compute_lipschitz_constants(SimpleNet().to(device), images, labels, nn.CrossEntropyLoss())
    print(f"Gradient Lipschitz constant: {grad_lipschitz:.2f}")
    print(f"Hessian Lipschitz constant: {hessian_lipschitz:.2f}")

    # Compare different optimizers
    optimizers = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'RMSProp': optim.RMSprop,
        'SmoothnessAware': SmoothnessAwareOptimizer
    }

    for name, optimizer_class in optimizers.items():
        print(f"Training with {name} optimizer")
        loss_history = train_with_optimizer(SimpleNet, train_loader, nn.CrossEntropyLoss(), optimizer_class, learning_rate, num_epochs)
        plt.plot(loss_history, label=name)

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss with Different Optimizers')
    plt.legend()
    plt.show()



