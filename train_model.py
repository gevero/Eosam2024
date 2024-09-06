import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# Define your custom dataset class (if necessary)
class MyDataset(data.Dataset):
    def __init__(self, csv_file):

        # dataframe from csv file
        df = pd.read_csv(csv_file)

        # groupd spectral data in single column
        df['spectra'] = df.values[:,5:].tolist()

        # drop leftover spectral data and rename columns
        df.drop(df.columns[5:-1], axis=1, inplace=True)
        df.columns = ['Lattice','Material','Thickness','Radius','Pitch','Spectra']

        self.data = df
        self.features = df[['Lattice','Material','Thickness','Radius','Pitch']]
        self.labels = df['Spectra']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Extract features and labels from the data
        features = torch.tensor(self.features.iloc[idx,].values.astype(np.float32))
        label = torch.tensor(self.labels.iloc[idx].astype(np.float32))
        return features, label

# Load the CSV data
csv_file = "your_data.csv"
dataset = MyDataset(csv_file)

# Split the dataset into training and validation sets using Scikit-learn
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders
batch_size = 32
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define your neural network model
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        # Define your model architecture here
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of your model
model = MyModel(input_dim=dataset[0][0].shape[0], output_dim=1)  # Adjust output_dim if needed

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Adjust loss function if needed
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create TensorBoard writer
writer = SummaryWriter("runs/your_experiment")

# Training and evaluation loop
num_epochs = 10
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(targets).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100.0 * train_correct / len(train_dataset)

    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / len(val_dataset)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Log metrics to TensorBoard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

# Close TensorBoard writer
writer.close()