from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import govars
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import os
import glob
import pickle

# Hyperparameters

num_epochs = 1_000
batch_size = 128
learning_rate = 1e-3

# Dataset dimensions
input_channels = govars.NUM_LAYERS - 1
board_size = 9
output_size = board_size**2 + 1 # 81 positions + 1 pass
test_size = 0.2

class ImprovedGoCNN(nn.Module):
    def __init__(self, input_channels):
        super(ImprovedGoCNN, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            self._create_residual_block(256) for _ in range(6)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )
        
    def _create_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        x = self.initial_conv(x)
        
        # Residual connections
        for block in self.residual_blocks:
            identity = x
            x = block(x)
            x += identity
            x = torch.relu(x)
            
        return self.policy_head(x)

def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=10,
    grad_clip=1.0,
    device=None
):
    """
    Training loop for Go CNN model with improved monitoring and stability.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization factor
        patience: Number of epochs to wait before early stopping
        grad_clip: Maximum norm of gradients
        device: Device to run on (will auto-detect if None)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # We're tracking accuracy
        factor=0.5,  # Halve the learning rate
        patience=5,
        verbose=True
    )
    
    # Training metrics tracking
    best_accuracy = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'learning_rates': []
    }
    
    # Normalize features
    def normalize_features(loader):
        features = []
        for batch_features, _ in loader:
            features.append(batch_features)
        features = torch.cat(features, dim=0)
        mean = features.mean()
        std = features.std()
        normalized_features = (features - mean) / (std + 1e-8)  # Add epsilon for numerical stability
        return normalized_features, mean, std
   
    train_features, mean, std = normalize_features(train_loader)
    test_features = []
    for batch_features, _ in test_loader:
        normalized_batch = (batch_features - mean) / (std + 1e-8)
        test_features.append(normalized_batch)
    test_features = torch.cat(test_features, dim=0)
        
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Convert one-hot to indices
            batch_labels = torch.argmax(batch_labels, dim=1)
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                batch_labels = torch.argmax(batch_labels, dim=1)
                
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_labels).item()
                
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
        
        avg_val_loss = val_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        # Update learning rate
        scheduler.step(accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['accuracy'].append(accuracy)
        history['learning_rates'].append(current_lr)
        
        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Learning Rate: {current_lr:.2e}\n')
        
        # Save best model and check for early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'history': history
            }, 'best_model.pth')
            print(f'New best model saved with accuracy: {accuracy:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return history

    
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

model = ImprovedGoCNN(input_channels=input_channels)

# Convert to PyTorch tensors
train_features = torch.tensor(train_features)
train_labels = torch.tensor(train_labels)
test_features = torch.tensor(test_features)
test_labels = torch.tensor(test_labels)

# Create data loader
train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=batch_size, shuffle=False)

train_model(model=model, train_loader=train_loader, test_loader=test_loader)