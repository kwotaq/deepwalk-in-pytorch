from random import choice

import matplotlib.pyplot as plt
import networkx as nx
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch import optim, tensor

class DeepWalk(nn.Module):
    def __init__(self, embedding_dim, graph=nx.karate_club_graph()):
        super().__init__()
        self.graph = graph
        num_nodes = graph.number_of_nodes()
        self.encode = nn.Embedding(num_nodes, embedding_dim)
        self.decode = nn.Linear(embedding_dim, num_nodes)

    def forward(self, inputs):
        embeds = self.encode(inputs).mean(0)
        return self.decode(embeds)

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

def get_integer_input(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

use_default = input("Do you want to run with default parameters? (yes/no): ").strip().lower()

if use_default == 'yes':
    walk_length = 8  # Default values
    max_num_walks = 100
    window_size = 5
    embedding_dim = 16
    epochs = 50
    lr = 0.001
    graph = nx.karate_club_graph()
else:
    walk_length = get_integer_input("Enter walk length: ", min_value=1)
    max_num_walks = get_integer_input("Enter max number of walks: ", min_value=1)
    window_size = get_integer_input("Enter window size: ", min_value=1, max_value=walk_length)
    embedding_dim = get_integer_input("Enter embedding dimensions: ", min_value=1)
    epochs = get_integer_input("Enter number of epochs: ", min_value=1)
    lr = get_integer_input("Enter learning rate: ", min_value=0.0001, max_value=0.1)
    graph =  nx.read_edgelist("data.txt", nodetype = int)

print("Running DeepWalk model with parameters:")
print(f"Walk Length: {walk_length}")
print(f"Max Number of Walks: {max_num_walks}")
print(f"Window Size: {window_size}")
print(f"Embedding Dimensions: {embedding_dim}")
print(f"Epochs: {epochs}")
print(f"Learning Rate: {lr}")

model = DeepWalk(embedding_dim)
model.train_model(window_size, lr, epochs, walk_length, max_num_walks)

# Extract embeddings from the model for use
embeddings = model.get_embeddings().numpy()

# Retrieve labels from embeddings using k-means
n_clusters = 2  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Train TSNE
tsne = TSNE(n_components=2,
            learning_rate='auto',
            init='pca',
            random_state=0).fit_transform(embeddings)

# Plot TSNE using labels created by k-means
plt.figure(figsize=(6, 6))
plt.scatter(tsne[:, 0], tsne[:, 1], s=100, c=labels, cmap="coolwarm")
plt.show()
