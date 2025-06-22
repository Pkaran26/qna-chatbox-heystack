# Import necessary Haystack components
from haystack import Document, Pipeline, component
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ChatMessage # Import ChatMessage for chat generators

from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

import os

# --- Important Setup for DeepSeek and Ollama ---
# For this script to work with DeepSeek, you need to:
# 1. Install Ollama: Go to https://ollama.com/ and download/install Ollama for your OS.
# 2. Pull a DeepSeek model using Ollama. For example, in your terminal:
#    ollama pull deepseek-r1:1.5b  # Pull the correct model tag you intend to use
#    You can find other DeepSeek models on Ollama's website or DeepSeek's GitHub.
# 3. Ensure the Ollama server is running (it usually starts automatically after installation).
# 4. No GOOGLE_API_KEY is needed if you are using an Ollama-hosted model.
# 5. Make sure to install the Haystack Ollama integration:
#    pip install ollama-haystack  # Corrected package name
# 6. For Milvus, install the Milvus Haystack integration:
#    pip install pymilvus milvus-haystack
# 7. For local Milvus Lite, you don't need a separate server, it uses a local file.
#    For a Milvus server (Docker/cloud), ensure it's running and accessible.

# --- Custom Component for Type Conversion ---
@component
class StringToChatMessageConverter:
    """
    A Haystack component to convert a single string prompt into a list of ChatMessage objects.
    This is necessary because PromptBuilder outputs a string, but ChatGenerators expect a list of ChatMessage.
    """
    @component.output_types(messages=list[ChatMessage])
    def run(self, prompt: str):
        # Wraps the input string into a single user ChatMessage
        return {"messages": [ChatMessage.from_user(prompt)]}

# --- Custom Component to Pass Through Documents and Expose Them ---
@component
class DocumentPassthrough:
    """
    A Haystack component to simply pass through documents.
    This is used to make the documents available to downstream components.
    """
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        return {"documents": documents}


# --- Read content from crawled_documents.txt ---
crawled_document_path = "crawled_documents.txt"
if os.path.exists(crawled_document_path):
    with open(crawled_document_path, "r", encoding="utf-8") as f:
        document_content = f.read()
    print(f"Successfully read content from {crawled_document_path}")
else:
    print(f"Error: {crawled_document_path} not found.")
    print("Please make sure 'crawled_documents.txt' exists in the same directory as the script.")
    # Exit or handle the error appropriately if the file is crucial
    document_content = "" # Initialize with empty string to avoid errors later


# --- Step 1: Initialize Document Store ---
# Using MilvusDocumentStore
# For Milvus Lite (local file-based), set uri to a local file path.
# For a Milvus server (Docker/remote), set uri to its address (e.g., "http://localhost:19530").
document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus_data.db"},  # Milvus Lite: Stores data in 'milvus_data.db' file
    # Or for a Milvus server:
    # connection_args={"uri": "http://localhost:19530"},
    drop_old=True,  # Drops existing collection with the same name if it exists (useful for fresh starts)
    # collection_name="haystack_docs" # You can specify a collection name, defaults to "haystack_collection"
)
print("MilvusDocumentStore initialized.")

# --- Step 2: Prepare Documents for Indexing Pipeline ---
# Use the content read from the file
documents = [Document(content=document_content, meta={"source": crawled_document_path})]

document_splitter = DocumentSplitter(
    split_by="passage",
    split_length=1000,
    split_overlap=200,
)
split_documents = document_splitter.run(documents=documents)["documents"]
print(f"Split document into {len(split_documents)} chunks.")

# --- Step 3: Set up Indexing Pipeline ---
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("document_embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
indexing_pipeline.add_component("document_writer", DocumentWriter(document_store=document_store))

indexing_pipeline.connect("document_embedder.documents", "document_writer.documents")

print("Starting document indexing...")
indexing_pipeline.run({"document_embedder": {"documents": split_documents}})
print("Document indexing complete.")
print(f"Number of documents in Milvus: {document_store.count_documents()}")

# --- Step 4: Set up Query Pipeline (RAG Pipeline) ---
query_pipeline = Pipeline()

query_pipeline.add_component("query_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))

# NEW: Use MilvusEmbeddingRetriever
query_pipeline.add_component("retriever", MilvusEmbeddingRetriever(document_store=document_store))

query_pipeline.add_component("doc_passthrough", DocumentPassthrough())

template = """
Given the following context, answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""
query_pipeline.add_component("prompt_builder", PromptBuilder(template=template, required_variables=["documents", "question"]))

query_pipeline.add_component("string_to_chat_message_converter", StringToChatMessageConverter())

query_pipeline.add_component("llm", OllamaChatGenerator(model="deepseek-r1:1.5b"))

query_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "doc_passthrough.documents")
query_pipeline.connect("doc_passthrough.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder.prompt", "string_to_chat_message_converter.prompt")
query_pipeline.connect("string_to_chat_message_converter.messages", "llm.messages")
print("Haystack RAG query pipeline set up with Milvus.")

# --- Step 5: Interact with the Chatbot ---
print("\n--- Q&A Chatbot Ready (using Haystack and DeepSeek via Ollama, with Milvus) ---")
print(f"Ask questions about the content from '{crawled_document_path}'.")
print("Type 'exit' to quit.")

while True:
    query = input("\nYour question: ")
    if query.lower() == 'exit':
        print("Exiting chatbot. Goodbye!")
        break

    if not query.strip():
        print("Please enter a question.")
        continue

    try:
        response = query_pipeline.run(
            {
                "query_embedder": {"text": query + 'with document url'},
                "prompt_builder": {"question": query + 'with document url'},
            }
        )

        answer = response["llm"]["replies"][0].text

        print(f"\nChatbot Answer: {answer}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure Ollama is installed and running, and the DeepSeek model is pulled.")
        print("Also ensure Milvus is running or that 'milvus_data.db' is writable for Milvus Lite.")