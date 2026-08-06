import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch_geometric.nn as gnn

# Step 1: Load dataset from the provided Excel file
train_data = pd.read_excel(r"D:\4 TH YEAR\FINAL YEAR PRO\GrapesDataset\grape_features_train.xlsx")

# Step 2: Extract features (X_train) and labels (y_train)
# Features are all columns except 'label'
features = train_data[['contrast_score', 'homogeneity_score', 'area', 'perimeter', 'eccentricity', 'score']].values
X_train = torch.tensor(features, dtype=torch.float)

# Ensure that the labels are in integer format (if not already)
y_train = torch.tensor(pd.to_numeric(train_data['label'], errors='coerce').fillna(0).astype(int).values, dtype=torch.long)

# Step 3: Create edge_index using KNN (or your own method for graph creation)
knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')  # You can change k to adjust the number of neighbors
knn.fit(X_train)
neighbors = knn.kneighbors(X_train)[1]  # Get indices of the k-nearest neighbors

# Create edge_index (pairs of connected nodes)
edge_index = []
for i in range(len(X_train)):
    for neighbor in neighbors[i]:
        edge_index.append([i, neighbor])  # Connect node i to its neighbor
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Convert to tensor

# Step 4: Prepare the Data object for PyTorch Geometric (GNN)
data = Data(x=X_train, edge_index=edge_index, y=y_train)

# Step 5: Define the GNN model using GCNConv layers
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = gnn.GCNConv(input_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Step 6: Initialize the model, loss function, and optimizer
model = GNNModel(input_dim=X_train.shape[1], hidden_dim=64, output_dim=3)  # Assuming 3 classes for classification
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 7: Training loop
def train(model, data, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)  # Forward pass
        loss = criterion(out, data.y)  # Compute the loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update the weights
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Step 8: Train the model
train(model, data, criterion, optimizer)

# Step 9: Save the trained model (gnn_model.pth)
torch.save(model.state_dict(), "gnn_model.pth")
print("Model saved as 'gnn_model.pth'")

# Step 10: Evaluate the model
def evaluate(model, data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to compute gradients during evaluation
        out = model(data)  # Forward pass
        _, pred = out.max(dim=1)  # Get predicted labels (class with max probability)
        correct = (pred == data.y).sum().item()  # Compare with true labels
        accuracy = correct / data.num_nodes  # Calculate accuracy
        print(f'Accuracy: {accuracy * 100:.2f}%')

# Call evaluation after training
evaluate(model, data)


