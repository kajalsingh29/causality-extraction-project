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
query_text = "Iontophoresis of << +/- propranolol >>, whose serotonergic actions include antagonism and partial agonism at [[ 5-HT1 ]] receptors, also increased serotonin and decreased firing (n=4)."
query_embedding = np.array(embedding_function.embed_query(query_text))  # Convert to NumPy array
print(query_embedding)
# Load the Chroma vector store
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Perform similarity search
results = db.similarity_search(query_text, k=100)

# Extract embeddings
stored_embeddings = [np.array(embedding_function.embed_query(result.page_content)) for result in results]  # Convert to NumPy arrays

all_embeddings = stored_embeddings + [query_embedding]  # Combine stored and query embeddings
labels = ['Stored'] * len(stored_embeddings) + ['Query']

# Convert to numpy array for compatibility with t-SNE
all_embeddings_array = np.array(all_embeddings)

# Perform dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced_embeddings = tsne.fit_transform(all_embeddings_array)

# Split the reduced embeddings back into stored and query embeddings
reduced_stored_embeddings = reduced_embeddings[:-1]  # All but the last one
reduced_query_embedding = reduced_embeddings[-1]  # The last one

# Calculate Euclidean distances in reduced space
distances = [np.linalg.norm(reduced_query_embedding - emb) for emb in reduced_stored_embeddings]

# Retrieve the indices of the top 5 closest embeddings
top_5_indices = np.argsort(distances)[:5]

# Retrieve top 5 results based on the indices
top_5_results = [results[i] for i in top_5_indices]

# Highlight the embeddings for visualization
reduced_top_5_embeddings = [reduced_stored_embeddings[i] for i in top_5_indices]

import seaborn as sns
import matplotlib.pyplot as plt
# Set Seaborn style for better aesthetics
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")

# Define color and marker properties
query_color = "purple"
retrieved_color = "pink"
non_retrieved_color = "black"

# Separate embeddings for plotting
non_retrieved_embeddings = np.delete(reduced_stored_embeddings, top_5_indices, axis=0)  # Non-retrieved embeddings
retrieved_embeddings = np.array([reduced_stored_embeddings[i] for i in top_5_indices])  # Retrieved embeddings

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot non-retrieved embeddings
ax.scatter(
    non_retrieved_embeddings[:, 0],
    non_retrieved_embeddings[:, 1],
    color=non_retrieved_color,
    alpha=0.6,
    marker='x',
    label="Non-retrieved text chunk embedding"
)

# Plot retrieved embeddings
ax.scatter(
    retrieved_embeddings[:, 0],
    retrieved_embeddings[:, 1],
    color=retrieved_color,
    alpha=0.8,
    marker='x',
    label="Retrieved text chunk embedding"
)

# Plot the query embedding
ax.scatter(
    reduced_query_embedding[0],
    reduced_query_embedding[1],
    color=query_color,
    edgecolor='black',
    s=80,  # Reduced size for better visualization
    label="Question embedding",
    zorder=5  # Bring to the front
)


# Add a dashed circle around the query embedding
max_distance = max([np.linalg.norm(reduced_query_embedding - retrieved) for retrieved in retrieved_embeddings])

# Create the circle
circle = plt.Circle(
    reduced_query_embedding, 
    max_distance, 
    color="grey", 
    fill=False, 
    linestyle="--", 
    linewidth=1.5, 
    label="Radius to Retrieved"
)

# Add the circle to the plot
ax.add_artist(circle)

# Set equal scaling of x and y axes to avoid distortion
ax.set_aspect('equal', adjustable='box')

# Add legend
ax.legend(
    fontsize=8,  # Adjust the font size
    loc="best",   # Best location based on plot content
    frameon=False,  # Remove background
    markerscale=1,  # Reduce the size of the markers in the legend (default is 1)
    handlelength=1.5,  # Length of the legend handles
    handleheight=1.5,
    labelspacing=1.5,  # Spacing between legend labels # Add a title to the legend (optional)
    title_fontsize=14,  # Set the font size for the legend title
)

# Add labels and title
ax.set_title("Embedding Space Visualization", fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
ax.set_ylabel("t-SNE Dimension 2", fontsize=12)

# Add grid lines for better readability
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Ensure layout is tight for better presentation
plt.tight_layout()

# Show the plot
plt.show()
