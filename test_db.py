from langchain.vectorstores.chroma import Chroma
from get_embeddings_function import get_embedding_function

CHROMA_PATH = "chroma"

# Initialize the Chroma database
db = Chroma(
    persist_directory=CHROMA_PATH, 
    embedding_function=get_embedding_function()
)

# Retrieve the document by ID
document_id = "element_100"  # Replace with your actual ID
result = db.get(ids=[document_id])  # Make sure to pass a list of IDs

# Check if the document was found and print it
if result and 'documents' in result:
    # Loop through the results (in case of multiple documents)
    for idx in range(len(result['documents'])):
        print(f"📄 Retrieved document {idx + 1}:")
        print("Content:", result['documents'][idx])
        print("Metadata:")
        print("  ID:", result['metadatas'][idx]['id'])
        print("  Source:", result['metadatas'][idx]['source'])
        print("  Entities:", result['metadatas'][idx]['entities'])
        print("  Relations:", result['metadatas'][idx]['relations'])
else:
    print(f"No document found with ID: {document_id}")
