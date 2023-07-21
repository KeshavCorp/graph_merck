# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:34:38 2023

@author: Kedar Kanhere
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt


# Function to filter the similarity matrix with thresholds
def filter_similarity_matrix(similarity_matrix, lower_threshold=0, upper_threshold=1):
    filtered_similarity_matrix = similarity_matrix.copy()
    filtered_similarity_matrix[filtered_similarity_matrix < lower_threshold] = 0
    filtered_similarity_matrix[filtered_similarity_matrix > upper_threshold] = 0
    return filtered_similarity_matrix

# Set the thresholds (default to 0 and 1 if not specified)
lower_threshold = 0.75
upper_threshold = 1


############## Sentence Transformer to Encode <You can try different things in here 
############## like tf idf word2vec etc. but transformers will give best results.

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the data from the CSV file
df = pd.read_csv('assets_data_prod.csv')


df['combined'] = df['short_description'].astype(str) + " " + df['long_description'].astype(str) + " "+ df['dataset'].astype(str) + " "+ df['therapeutic_area'].astype(str) + " "+ df['type'].astype(str) + " " + df['type'].astype(str) + " "+ df['source_url_provider'].astype(str)


# Get the 'paragraphs' column
paragraphs = df['combined']



######################### Cosine ##############################################

# Compute embeddings for each paragraph
embeddings = model.encode(paragraphs)

# Compute cosine similarity for all pairs of paragraphs
similarity_matrix = cosine_similarity(embeddings)

# Now, similarity_matrix[i][j] is the similarity score between paragraphs[i] and paragraphs[j]
print(similarity_matrix)





# Filter the similarity matrix
filtered_similarity_matrix = filter_similarity_matrix(similarity_matrix, lower_threshold, upper_threshold)

pd.DataFrame(filtered_similarity_matrix).to_csv("Similarity_matrix.csv")
########################### Jaccard ###########################################

embedding_sets = [set(map(str, embedding)) for embedding in embeddings]

# Function to calculate Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set2))
    union = len(set1) + len(set2) - intersection
    return float(intersection) / union

# Compute Jaccard similarity for all pairs of paragraphs
jaccard_similarities = [[jaccard_similarity(x, y) for y in embedding_sets] for x in embedding_sets]

# Now, jaccard_similarities[i][j] is the Jaccard similarity between paragraphs[i] and paragraphs[j]
print(jaccard_similarities)





# Create a graph
graph = nx.Graph()

# Add nodes to the graph
num_texts = len(df['combined'].tolist())
for i in range(num_texts):
    graph.add_node(i)
    
    
    
for i in range(num_texts):
    for j in range(i + 1, num_texts):
        similarity_score = filtered_similarity_matrix[i, j]
        if similarity_score != 0:
            graph.add_edge(i, j, weight=similarity_score)

# Get a list of nodes that are connected to at least one other node
connected_nodes = [node for node in graph.nodes() if len(list(graph.neighbors(node))) > 0]


# Create a subgraph with only the connected nodes
subgraph = graph.subgraph(connected_nodes)

# Draw the subgraph using Fruchterman-Reingold layout with custom edge lengths
pos = nx.spring_layout(subgraph, k=0.3, iterations=50)

# Get the edge weights for visualization
edge_weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]

plt.figure(figsize=(12, 10))
nx.draw_networkx(subgraph, pos, labels={i: f'Row {i}' for i in subgraph.nodes()},
                 node_color='lightblue', node_size=500, font_size=10,
                 edge_color=edge_weights, edge_cmap=plt.cm.Blues, width=2.5, alpha=0.7)
nx.draw_networkx_edge_labels(subgraph, pos, edge_labels={(u, v): f'{subgraph[u][v]["weight"]:.2f}' for u, v in subgraph.edges()},
                             font_size=8, label_pos=0.3)
plt.title('Text Similarity Graph (Filtered)')
plt.axis('off')
plt.show()
