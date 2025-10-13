import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
collection_name = "impot_collection"
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
def main():
  try:
    if client.get_collection(name=collection_name):
      print(f"Collection '{collection_name}' already exists.")
    else:
      collection = client.create_collection(
        name= collection_name,
        metadata={"description":"A collection of information about taxes"},
        configuration={
          "hnsw": {
            "space":"cosine",
            # Maximum Connections Range: Typically 4-64 default : 16
            "M": 32, 
            #  Construction-time Search Width Range: Usually 100-800
            "ef_construction": 400,  # Default is 200
            # Query-time Search Width Range: Usually 10-200
            "ef": 100,  # Default is 10
            # Maximum number of elements (vectors) the index can hold
            "max_elements": 10000  # Default is 1000000
            },
          "embedding_function": ef
        }
      )

  except Exception as error:  # Catch any errors and log them to the console
    print(f"Error: {error}")
    