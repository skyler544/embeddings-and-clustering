from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define the functional requirements for EarlyBird system
requirements_file = "early-bird-requirements.txt"
with open(requirements_file, "r") as file:
    requirements = [line.strip() for line in file if line.strip()]  # Remove empty lines and strip spaces

# Step 2: Convert Requirements into Embeddings using Sentence-BERT
print("Generating embeddings for the requirements...")

# Pick a model from:
# https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
# These sentence transformer models are luckily much smaller than full LLMs
model = SentenceTransformer('all-mpnet-base-v2')

# Generate embeddings for each requirement
embeddings = model.encode(requirements)

# Print the shape of the embeddings to confirm they are generated correctly
print("Embedding Shape:", np.array(embeddings).shape)

# Step 3: Store Embeddings in FAISS for similarity search and clustering

# Convert embeddings to NumPy array (required by FAISS)
embeddings_np = np.array(embeddings).astype('float32')

# Create a FAISS index (using L2 distance for clustering)
index = faiss.IndexFlatL2(embeddings_np.shape[1])

# Add the embeddings to the FAISS index
index.add(embeddings_np)

# Confirm that the embeddings are added to the index
print(f"Number of embeddings in FAISS index: {index.ntotal}")

# Step 4: Perform K-Means Clustering to group the requirements

# Set the number of clusters (adjust this based on your needs)
# TODO try changing the number of clusters
n_clusters = 5

# TODO try other clustering strategies
# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(embeddings_np)

# Group requirements by clusters
clustered_requirements = {i: [] for i in range(n_clusters)}
for i, label in enumerate(labels):
    clustered_requirements[label].append(requirements[i])

# Step 5: Generate names for each cluster
# Get centroids of clusters
centroids = kmeans.cluster_centers_

# Find the most representative sentence for each cluster (closest to the centroid)
cluster_names = {}
for cluster_idx in range(n_clusters):
    # Get embeddings of items in the current cluster
    cluster_items_embeddings = embeddings_np[labels == cluster_idx]

    # Compute similarity of each item to the centroid
    similarities = cosine_similarity([centroids[cluster_idx]], cluster_items_embeddings)

    # Find the most representative sentence (highest similarity)
    most_representative_idx = np.argmax(similarities)
    representative_sentence = requirements[np.where(labels == cluster_idx)[0][most_representative_idx]]

    # Use the representative sentence as the cluster name
    cluster_names[cluster_idx] = representative_sentence

# Step 6: Output the Clusters
print("\nClustered Requirements:")
for cluster, items in clustered_requirements.items():
    print(f"\nCluster {cluster + 1}: {cluster_names[cluster]}")
    for item in items:
        print(f"  - {item}")
