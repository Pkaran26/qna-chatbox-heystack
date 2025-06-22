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


# --- Sample Document (Replace with your actual document content) ---
sample_document_content = """
To create a Linux instance, follow these steps:

Navigate to Compute > Linux Instances.
Click on NEW LINUX INSTANCE button. Create Linux Instance
Choose an Availability Zone, which is the geographical region where your Instance will be deployed. Create Linux Instance
Select a VPC or VNF network from the Select Network Dropdown and, select the appropriate tier listed in Select Network.
note
To add a Linux Instance to a VPC or VNF, you need to have a VPC or VNF configured with at least one tier.

Select the OS Image to run on your Instance.
Select the Compute Pack from the available compute collections.
Select the Root Disk from the available options.
Select the option to Protect this Instance. Compute Pack
In Choose Instant Apps, select the available applications. To Verify/Login into your selected database, refer to App Overlays. Instant Apps
Choose an Authentication Method:
Use SSH key pair: To view all the SSH key pairs present in your account, click the Use SSH key pair option. If your account doesn’t have any SSH key pair, then you can click the Generate a new key pair or upload the key pair by clicking the Upload a key pair option.
Use root user password: On selecting Use root user password, the Also email me the password option is displayed. If you select this option, the password, along with the details, for instance, will be emailed to your registered email ID.
In the Name Your Linux Instance field, enter the desired name for your Linux Instance. The Instance name contains alphanumeric characters, underscore, dots and hyphens only.
Verify the Estimated Cost of your Linux Instance based on the chosen specifications from the Summary and Estimated Costs Section (Here, both Hourly and Monthly Prices summary are displayed). Summary
Select the I have read and agreed to the End User License Agreement and Privacy Policy option.
Choose the BUY HOURLY or BUY MONTHLY option. A confirmation window appears and the price summary will be displayed along with the discount codes if you have any in your account.
You can apply any of the discount codes listed by clicking on the APPLY button.
You can also remove the applied discount code by clicking on the REMOVE button.
You can cancel this action by clicking on the CANCEL button. Discount Codes
Click CONFIRM to create the Linux Instance.
note
It might take up to 5-8 minutes for the Linux instance to get created. You may use the Cloud Console during this time, but it is advised that you do not refresh the browser window.

Once ready, you get notified of this purchase on your registered email ID. To access the newly created Linux Instances, navigate to Compute > Linux Instances on the main navigation panel.
"""

# --- Step 1: Initialize Document Store ---
# Using MilvusDocumentStore
# For Milvus Lite (local file-based), set uri to a local file path.
# For a Milvus server (Docker/remote), set uri to its address (e.g., "http://localhost:19530").
document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus_data.db"},  # Milvus Lite: Stores data in 'milvus_data.db' file
    # Or for a Milvus server:
    # connection_args={"uri": "http://localhost:19530"},
    drop_old=True,  # Drops existing collection with the same name if it exists (useful for fresh starts)
    # REMOVE THIS LINE: embedding_dim=384, # This is causing the TypeError
    # collection_name="haystack_docs" # You can specify a collection name, defaults to "haystack_collection"
)
print("MilvusDocumentStore initialized.")

# --- Step 2: Prepare Documents for Indexing Pipeline ---
documents = [Document(content=sample_document_content, meta={"source": "sample_document"})]

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
print("Ask questions about the Linux instance creation process (or your document content).")
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
                "query_embedder": {"text": query},
                "prompt_builder": {"question": query},
            }
        )

        answer = response["llm"]["replies"][0]

        print(f"\nChatbot Answer: {answer}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure Ollama is installed and running, and the DeepSeek model is pulled.")
        print("Also ensure Milvus is running or that 'milvus_data.db' is writable for Milvus Lite.")