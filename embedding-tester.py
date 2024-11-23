from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.cluster import KMeans

# Step 1: Define the functional requirements for EarlyBird system
# TODO setup the requirements in a more super way
requirements = [
    "Allow users to create an account",
    "Allow users to search for available flights",
    "Allow users to book a flight",
    "Send notifications to users regarding flight updates",
    "Process payment transactions",
    "Allow users to cancel a flight",
    "Allow users to view their booking history",
    "Integrate payment gateways for credit card processing",
    "Manage flight availability and booking capacity",
    "Generate booking confirmation emails"
]

# Step 2: Convert Requirements into Embeddings using Sentence-BERT
print("Generating embeddings for the requirements...")

# Initialize the Sentence-BERT model
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('paraphrase-distilroberta-base-v1')
model = SentenceTransformer('stsb-roberta-large')

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
n_clusters = 3

# TODO try other clustering strategies
# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(embeddings_np)

# Group requirements by clusters
clustered_requirements = {i: [] for i in range(n_clusters)}
for i, label in enumerate(labels):
    clustered_requirements[label].append(requirements[i])

# Step 5: Output the Clusters and Name the Components
print("\nClustered Requirements and Suggested Component Names:")
for cluster, items in clustered_requirements.items():
    print(f"\nCluster {cluster + 1}:")
    for item in items:
        print(f"  - {item}")

# Suggested Component Names based on clusters (just an example)
print("\nSuggested Component Names for each Cluster:")
for cluster, items in clustered_requirements.items():
    if cluster == 0:
        print(f"Component for Cluster {cluster + 1}: User Management")
    elif cluster == 1:
        print(f"Component for Cluster {cluster + 1}: Flight Booking & Search")
    elif cluster == 2:
        print(f"Component for Cluster {cluster + 1}: Payments & Notifications")
