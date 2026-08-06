# In Louvain_clustering.py
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

def calculate_similarity(vec1, vec2):
    """Calculate cosine similarity between two PyTorch tensors."""
    vec1 = vec1.unsqueeze(0).numpy()
    vec2 = vec2.unsqueeze(0).numpy()
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return similarity

def run_Louvain_clustering():
    # Load processed data
    train_data = pd.read_excel("D:/4 TH YEAR/FINAL YEAR PRO/GrapesDataset/grape_features_train.xlsx")
    test_data = pd.read_excel("D:/4 TH YEAR/FINAL YEAR PRO/GrapesDataset/grape_features_test.xlsx")

    # Extract features and labels
    X_train = train_data[["contrast_score", "homogeneity_score", "area", "perimeter", "eccentricity"]].values
    y_train = train_data["label"].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Build KNN graph
    knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knn.fit(X_train)
    neighbors = knn.kneighbors(X_train)[1]

    edge_index = []
    for i in range(len(X_train)):
        for neighbor in neighbors[i]:
            edge_index.append([i, neighbor])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=X_train, edge_index=edge_index, y=y_train)

    # Create graph
    G = nx.Graph()
    for i, feature in enumerate(X_train):
        G.add_node(i, feature=feature)

    threshold = 0.8
    for i in range(len(X_train)):
        for j in range(i + 1, len(X_train)):
            if calculate_similarity(X_train[i], X_train[j]) > threshold:
                G.add_edge(i, j)

    # Louvain clustering
    partition = community_louvain.best_partition(G)

    # Visualization
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    community_colors = [partition[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=community_colors, cmap=plt.cm.tab10, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.0, edge_color='gray')
    nodes_to_label = [0, 50, 100, 150, 200, 250]
    labels = {i: f"Node {i}" for i in nodes_to_label}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', font_color='black')

    plt.title("Graph Visualization with Louvain Communities (Selected Nodes Highlighted)", fontsize=16)
    plt.axis('off')
    plt.show()

    print("Community Assignments (Node -> Community):")
    for node, community in partition.items():
        print(f"Node {node}: Community {community}")
