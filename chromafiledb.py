import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read document
with open(r"C:\Users\GSA15\OneDrive\Desktop\gen ai.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = splitter.split_text(text)

# Create ChromaDB
client = chromadb.PersistentClient(path="vectordb")
collection = client.get_or_create_collection(name="documents")

# Store embeddings
for i, chunk in enumerate(chunks):
    embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=chunk
    )["embedding"]

    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embedding]
    )

print("Documents indexed!")

# Ask a question
question = input("Ask a question: ")

# Embed the question
query_embedding = ollama.embeddings(
    model="nomic-embed-text",
    prompt=question
)["embedding"]

# Retrieve relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

context = "\n".join(results["documents"][0])

# Send context to Ollama
prompt = f"""
Answer using the context below.

Context:
{context}

Question:
{question}
"""

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nAnswer:\n")
print(response["message"]["content"])