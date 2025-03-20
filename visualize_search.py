from langchain.vectorstores import Chroma
from get_embeddings_function import get_embedding_function
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the Chroma database
CHROMA_PATH = "chroma_chemprot"

# Initialize the embedding function
embedding_function = get_embedding_function()

# Input query and generate embedding
query_text = "<< Androgen >> antagonistic effect of estramustine phosphate (EMP) metabolites on wild-type and mutated [[ androgen receptor ]]"
query_embedding = np.array(embedding_function.embed_query(query_text))  # Convert to NumPy array
print(query_embedding)
# Load the Chroma vector store
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Perform similarity search
results = db.similarity_search(query_text, k=100)

# Extract embeddings
stored_embeddings = [np.array(embedding_function.embed_query(result.page_content)) for result in results]  # Convert to NumPy arrays


distances = [np.linalg.norm(query_embedding - emb) for emb in stored_embeddings]

top_5_indices = np.argsort(distances)[:5]  # Indices of top 5 closest embeddings


# Sort by cosine similarity (highest similarity first) and retrieve the top 5 results


# Retrieve top 5 results based on the sorted indices
top_5_results = [results[i] for i in top_5_indices]


# Combine embeddings for visualization
all_embeddings = stored_embeddings + [query_embedding]
labels = ['Stored'] * len(stored_embeddings) + ['Query']

# Dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced_embeddings = tsne.fit_transform(np.array(all_embeddings))

# Visualize embeddings
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:-1, 0], reduced_embeddings[:-1, 1], alpha=0.5, label='Stored Embeddings')
plt.scatter(reduced_embeddings[-1, 0], reduced_embeddings[-1, 1], color='red', label='Query Embedding')

# Highlight top 5 results
for i in top_5_indices:
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], color='green', label='Top 5 Result')

# Draw a circle around the query embedding
query_point = reduced_embeddings[-1]
max_distance = max([np.linalg.norm(query_point - reduced_embeddings[i]) for i in top_5_indices])
circle = plt.Circle(query_point, max_distance, color='blue', fill=False, linestyle='--', label='Radius to Top 5')
plt.gca().add_artist(circle)

plt.legend()
plt.title('Embedding Space Visualization with Radius')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
