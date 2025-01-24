import os
from random import choice

import matplotlib.pyplot as plt
import networkx as nx
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch import optim, tensor

class DeepWalk(nn.Module):
    def __init__(self, embedding_dim, graph):
        super().__init__()
        self.graph = graph
        num_nodes = graph.number_of_nodes()
        self.encode = nn.Embedding(num_nodes, embedding_dim)
        self.decode = nn.Linear(embedding_dim, num_nodes)

    def forward(self, inputs):
        hidden = self.encode(inputs).mean(0)
        return self.decode(hidden)

    def get_embeddings(self):
        return self.decode.weight.detach()

    def random_walk(self, graph, start_node, walk_length):
        walk = [start_node]
        next_node = start_node
        for _ in range(walk_length):
            next_node = choice(list(graph.neighbors(next_node)))
            walk.append(next_node)

        return walk

    def get_training_data(self, graph, window, walk_length, walks):
        print("Creating training data through random walks..")
        window_side = (window - 1) // 2
        walks_table = []
        for n in graph.nodes():
            for _ in range(walks):
                walk = self.random_walk(graph, n, walk_length)
                for idx in range(len(walk) - window + 1): # Limit index so the end of the window reaches until the end of the array and no further
                    target = walk[idx + window_side]
                    context = walk[idx: idx + window_side] + walk[idx + window_side + 1: idx + window]
                    walks_table.append((context, target))

        print("Training data created.")
        return walks_table

    def train_model(self, window, lr, epochs, walk_length, walks):
        walks_table = self.get_training_data(self.graph, window, walk_length, walks)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr)

        print("Initializing training..")
        for epoch in range(epochs):
            running_loss = 0
            for context, target in walks_table:
                optimizer.zero_grad()
                pred = model(tensor(context))
                loss = loss_fn(pred, tensor(target))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                print(f"In epoch {epoch}, loss: {running_loss / len(walks_table):.3f}")

        print("Training finished.")

def get_integer_input(prompt, min_value=None, max_value=None, must_be_odd=False, default=None):
    print("Press Enter to skip.")
    while True:
        user_input = input(prompt).strip()
        if user_input == "":
            return default
        try:
            value = int(user_input)
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}.")
                continue
            if must_be_odd and value % 2 == 0:
                print("Value must be an odd number.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_graph_from_terminal(prompt, default):
    print("Press Enter to skip.")
    while True:
        file_path = input(prompt).strip()
        if file_path == "":
            return default
        if os.path.isfile(file_path):
            try:
                graph = nx.read_edgelist(file_path)  # Attempt to load as a networkx graph
                return graph
            except Exception:
                print("Invalid graph file. Please enter a valid NetworkX graph file.")
        else:
            print("Invalid file path. Please enter a valid path to an existing file.")

def get_yes_no_input(prompt):
    while True:
        user_input = input(prompt + " (yes/no): ").strip().lower()
        if user_input in ["yes", "no"]:
            return user_input
        print("Invalid input. Please enter 'yes' or 'no'.")

use_default = get_yes_no_input("Do you want to run with default parameters (default graph is Zachary Karate Club Graph)? (yes/no): ")

graph = nx.karate_club_graph()
walk_length = 10  # Default values
num_walks = 12
window_size = 5
embedding_dim = 32
epochs = 50
lr = 0.001

if use_default == 'no':
    graph = get_graph_from_terminal("Enter the path to the graph file: ", default=graph)
    walk_length = get_integer_input("Enter walk length: ", min_value=1, default=walk_length)
    num_walks = get_integer_input("Enter the number of walks: ", min_value=1, default=num_walks)
    window_size = get_integer_input("Enter window size: ", min_value=1, max_value=walk_length, must_be_odd=True, default=window_size)
    embedding_dim = get_integer_input("Enter embedding dimensions: ", min_value=1, default=embedding_dim)
    epochs = get_integer_input("Enter number of epochs: ", min_value=1, default=epochs)
    lr = get_integer_input("Enter learning rate: ", min_value=0.0001, max_value=0.1, default=lr)

print("Running DeepWalk model with parameters:")
print(f"Walk Length: {walk_length}")
print(f"Number of Walks: {num_walks}")
print(f"Window Size: {window_size}")
print(f"Embedding Dimensions: {embedding_dim}")
print(f"Epochs: {epochs}")
print(f"Learning Rate: {lr}")

# Initialize model
model = DeepWalk(embedding_dim, graph)
model.train_model(window_size, lr, epochs, walk_length, num_walks)

# Extract embeddings from the model for use
embeddings = model.get_embeddings().numpy()

n_clusters = 2  # Default number of clusters
use_default = get_yes_no_input("Do you want to plot the default amount of cluster labels (defaults to 2)? (yes/no): ")

if use_default == 'no':
    n_clusters = get_integer_input("Enter number of clusters: ", min_value=2, default=n_clusters)

# Retrieve labels from embeddings using k-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Train TSNE
tsne = TSNE(n_components=2,
            learning_rate='auto',
            init='pca',
            random_state=0).fit_transform(embeddings)

# Plot TSNE using labels created with k-means
plt.figure(figsize=(6, 6))
plt.scatter(tsne[:, 0], tsne[:, 1], s=100, c=labels, cmap="coolwarm")
title = f"Scatterplot of node embeddings for {n_clusters} clusters. \nParameters used: embedding dimensions: {embedding_dim}, number of walks: {num_walks}, walk length: {walk_length}, window size: {window_size}"
plt.title(title, loc='center', wrap=True)
plt.show()
