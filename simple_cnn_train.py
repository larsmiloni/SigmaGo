import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
import os
import pickle
import numpy as np
import govars

# Hyperparameters
num_epochs = 1_000
batch_size = 128
learning_rate = 1e-3

# Dataset dimensions
input_channels = govars.NUM_LAYERS - 1
board_size = 9
output_size = board_size**2 + 1 # 81 positions + 1 pass
test_size = 0.2

# Model definition
class GoCNN(nn.Module):
    def __init__(self):
        super(GoCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, output_size)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output without activation, will use CrossEntropyLoss which expects logits
        return x

# Load dataset
def load_features_labels(test_size: float):
    cwd = os.getcwd()
    pickleRoot = os.path.join(cwd, 'pickles')
    mixedPickleRoot = os.path.join(cwd, 'pickles_mixed')

    features_list, labels_list = [], []

    def loadPickle(pickleFile, features_list, labels_list):
        try:
            with open(pickleFile, 'rb') as f:
                print("Loading from", pickleFile)
                saved = pickle.load(f)
                datasetNew = saved['dataset'].astype('float32')
                labelsNew = saved['labels'].astype('float32')


                features_list.append(datasetNew)
                labels_list.append(labelsNew)

                print(f"Loaded data - Features shape: {datasetNew.shape}, Labels shape: {labelsNew.shape}")
        except Exception as e:
            print(f"Unable to load data from {pickleFile}: {e}")

    # Load all pickle files
    for pickleFile in glob.glob(os.path.join(pickleRoot, "*.pickle")):
        loadPickle(pickleFile, features_list, labels_list)
    for mixedPickleFile in glob.glob(os.path.join(mixedPickleRoot, "*.pickle")):
        loadPickle(mixedPickleFile, features_list, labels_list)

    # Concatenate lists into arrays
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Split into training and test sets
    data_size = len(features)
    test_size = int(test_size * data_size)
    train_size = data_size - test_size
    x_train, x_test = features[:train_size], features[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]

    print("Final shapes - x_train:", x_train.shape, "y_train:", y_train.shape,
          "x_test:", x_test.shape, "y_test:", y_test.shape)

    return x_train, y_train, x_test, y_test

train_features, train_labels, test_features, test_labels = load_features_labels(test_size=test_size)

# Convert to PyTorch tensors
train_features = torch.tensor(train_features)
train_labels = torch.tensor(train_labels)
test_features = torch.tensor(test_features)
test_labels = torch.tensor(test_labels)

# Create data loader
train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = GoCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_features, batch_labels in train_loader:
        # Move data to GPU if available
        if torch.cuda.is_available():
            batch_features, batch_labels = batch_features.cuda(), batch_labels.cuda()
            model = model.cuda()
            
        # Convert one-hot labels to class indices
        batch_labels = torch.argmax(batch_labels, dim=1)

        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_features.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Validation step
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            if torch.cuda.is_available():
                batch_features, batch_labels = batch_features.cuda(), batch_labels.cuda()
                
            # Convert one-hot labels to class indices for validation as well
            batch_labels = torch.argmax(batch_labels, dim=1)

            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    # Calculate and display test accuracy
    test_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%\n")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'go_cnn_model.pth')

print("Training complete and model saved.")
