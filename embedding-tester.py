from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Define the functional requirements for EarlyBird system
requirements_file = "early-bird-requirements.txt"
with open(requirements_file, "r") as file:
    requirements = [
        line.strip() for line in file if line.strip()
    ]  # Remove empty lines and strip spaces

# Step 2: Convert Requirements into Embeddings using Sentence-BERT
print("Generating embeddings for the requirements...")

# Pick a model from:
# https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
# These sentence transformer models are luckily much smaller than full LLMs
model = SentenceTransformer("all-mpnet-base-v2")

# Generate embeddings for each requirement
embeddings = model.encode(requirements)

# Print the shape of the embeddings to confirm they are generated correctly
print("Embedding Shape:", np.array(embeddings).shape)

# Step 3: Store Embeddings in FAISS for similarity search and clustering

# Convert embeddings to NumPy array (required by FAISS)
embeddings_np = np.array(embeddings).astype("float32")

# Create a FAISS index (using L2 distance for clustering)
index = faiss.IndexFlatL2(embeddings_np.shape[1])

# Add the embeddings to the FAISS index
index.add(embeddings_np)

# Confirm that the embeddings are added to the index
print(f"Number of embeddings in FAISS index: {index.ntotal}")

# Step 4: Perform K-Means Clustering to group the requirements

# Set the number of clusters (adjust this based on your needs)
n_clusters = 5

# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(embeddings_np)

# Group requirements by clusters
clustered_requirements = {i: [] for i in range(n_clusters)}
for i, label in enumerate(labels):
    clustered_requirements[label].append(requirements[i])


# Step 5: Determine the topic of each cluster
def extract_keywords(requirements, top_n=3):
    """Extract top keywords from a list of requirements using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(requirements)
    feature_names = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1  # Sum TF-IDF scores across all documents
    keywords = [feature_names[i] for i in scores.argsort()[-top_n:][::-1]]
    return ", ".join(keywords)


cluster_descriptions = {}
for cluster_idx, items in clustered_requirements.items():
    cluster_descriptions[cluster_idx] = extract_keywords(items, top_n=3)

# Step 6: Output the Clusters
print("\nClustered Requirements:")
for cluster, items in clustered_requirements.items():
    description = cluster_descriptions[cluster]
    print(f"\nCluster {cluster + 1}: {description.capitalize()}")
    for item in items:
        print(f"  - {item}")
