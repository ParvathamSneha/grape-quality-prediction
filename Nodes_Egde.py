import torch
from torch_geometric.data import Data
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from community import community_louvain

# Load processed data
train_data = pd.read_excel("D:/4 TH YEAR/FINAL YEAR PRO/GrapesDataset/grape_features_train.xlsx")
test_data = pd.read_excel("D:/4 TH YEAR/FINAL YEAR PRO/GrapesDataset/grape_features_test.xlsx")

# Extract features and labels
X_train = train_data[["contrast_score", "homogeneity_score", "area", "perimeter", "eccentricity"]].values
y_train = train_data["label"].values

# Encode labels if needed
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Convert features to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)

# Create edges using KNN (k-nearest neighbors)
knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
knn.fit(X_train)
neighbors = knn.kneighbors(X_train)[1]  # Find k-nearest neighbors for each data point

# Create edge indices
edge_index = []
for i in range(len(X_train)):
    for neighbor in neighbors[i]:
        edge_index.append([i, neighbor])  # Connect node i to its neighbor
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create graph data
data = Data(x=X_train, edge_index=edge_index, y=y_train)

# Define the similarity function (Cosine Similarity)
def calculate_similarity(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# Create a graph for visualization
G = nx.Graph()

# Add nodes to the graph
for i, feature in enumerate(X_train):
    G.add_node(i, feature=feature)  # Add nodes with features

# Add edges based on similarity threshold
threshold = 0.8
for i in range(len(X_train)):
    for j in range(i + 1, len(X_train)):
        if calculate_similarity(X_train[i], X_train[j]) > threshold:
            G.add_edge(i, j)

# Apply Louvain clustering
partition = community_louvain.best_partition(G)  # Community detection

# Function to calculate node degree
def calculate_node_degree(graph):
    return {node: degree for node, degree in graph.degree()}

# Calculate node degrees
node_degrees = calculate_node_degree(G)

# Calculate average feature value for each node
node_avg_feature_value = {}
for node in G.nodes():
    features = G.nodes[node]['feature']
    avg_feature_value = np.mean(features.numpy())  # Convert PyTorch tensor to numpy array for calculation
    node_avg_feature_value[node] = avg_feature_value

# Store edge similarities (weights)
edge_similarities = {}
for edge in G.edges():
    node1, node2 = edge
    similarity = calculate_similarity(X_train[node1], X_train[node2])  # Cosine similarity between features
    edge_similarities[edge] = similarity

# Visualization with Louvain communities
plt.figure(figsize=(14, 10))

# Adjust the layout
pos = nx.spring_layout(G, k=0.15, iterations=20)

# Color nodes based on Louvain communities
community_colors = [partition[node] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=100, node_color=community_colors, cmap=plt.cm.tab10, alpha=0.9)

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.0, edge_color='gray')

# Add labels for specified nodes only
nodes_to_label = [0, 50, 100, 150, 200, 250]  # Nodes to label
labels = {i: f"Node {i}" for i in nodes_to_label}
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', font_color='black')

# Set title and show plot
plt.title("Graph Visualization with Louvain Communities (Selected Nodes Highlighted)", fontsize=16)
plt.axis('off')  # Hide axes for better presentation
plt.show()

# Print Community Assignments and Properties
print("Community Assignments (Node -> Community):")
for node, community in partition.items():
    print(f"Node {node}: Community {community}")
    
print("Node Degrees:", node_degrees)
print("Node Average Feature Values:", node_avg_feature_value)
print("Edge Similarities:", edge_similarities)
