# Import necessary Haystack components
from haystack import Document, Pipeline, component
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
# Corrected import path for OllamaChatGenerator:
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ChatMessage # Import ChatMessage for chat generators

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
# This component is kept for conceptual clarity in the RAG chain,
# but its output won't be directly accessible at the top level of `pipeline.run()`
# in versions of Haystack that don't support `add_outputs` or `debug_outputs`
# as top-level parameters.
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
# InMemoryDocumentStore is good for quick starts and smaller datasets.
# For larger applications, consider persistent document stores like Weaviate, Pinecone, Elasticsearch, etc.
document_store = InMemoryDocumentStore()
print("InMemoryDocumentStore initialized.")

# --- Step 2: Prepare Documents for Indexing Pipeline ---
# Convert the raw text content into a Haystack Document object.
# In a real application, you'd load from files (e.g., PDF, TXT) using components like `FileToDocument`.
documents = [Document(content=sample_document_content, meta={"source": "sample_document"})]

# Split the document into smaller chunks.
# Updated DocumentSplitter initialization for Haystack 2.x API
document_splitter = DocumentSplitter(
    split_by="passage",  # Corrected: split_by must be one of the valid options (e.g., "passage", "sentence", "word")
    split_length=1000,
    split_overlap=200,
)
split_documents = document_splitter.run(documents=documents)["documents"]
print(f"Split document into {len(split_documents)} chunks.")

# --- Step 3: Set up Indexing Pipeline ---
# This pipeline processes your documents, embeds them, and writes them to the document store.
indexing_pipeline = Pipeline()
# Document Embedder to create embeddings for our documents.
# 'sentence-transformers/all-MiniLM-L6-v2' is a small, efficient model suitable for offline use.
indexing_pipeline.add_component("document_embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
indexing_pipeline.add_component("document_writer", DocumentWriter(document_store=document_store))

# Connect components in the indexing pipeline
indexing_pipeline.connect("document_embedder.documents", "document_writer.documents")

# Run the indexing pipeline to add documents to the store
print("Starting document indexing...")
indexing_pipeline.run({"document_embedder": {"documents": split_documents}})
print("Document indexing complete.")

# --- Step 4: Set up Query Pipeline (RAG Pipeline) ---
# This pipeline will take a query, retrieve relevant documents, and generate an answer.
query_pipeline = Pipeline() # No 'outputs' in constructor for this Haystack version

# Removed query_pipeline.add_outputs(...) as it's not supported in this Haystack version.
# This means `retrieved_documents` will not be directly accessible from the top-level
# `response` dictionary via `response["retrieved_docs"]`.
# The `doc_passthrough` component remains in the pipeline to ensure documents flow to
# the `prompt_builder`, but its output is consumed and not exposed at the top level.


# Text Embedder for the user's query. It must use the same model as the document embedder.
query_pipeline.add_component("query_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))

# Retriever to find relevant documents from the document store based on the query embedding.
query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))

# Add the passthrough component to make retriever's documents flow to prompt_builder
query_pipeline.add_component("doc_passthrough", DocumentPassthrough())

# Prompt Builder to construct the prompt for the LLM, including retrieved context.
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
query_pipeline.add_component("prompt_builder", PromptBuilder(template=template, required_variables=["documents", "question"])) # Added required_variables

# Add the new converter component to transform the string prompt into ChatMessage list
query_pipeline.add_component("string_to_chat_message_converter", StringToChatMessageConverter())

# LLM Generator using Ollama.
# Specify the model you pulled with Ollama (e.g., "deepseek-r1:1.5b").
# Make sure the Ollama server is running and the model is downloaded.
query_pipeline.add_component("llm", OllamaChatGenerator(model="deepseek-r1:1.5b")) # Changed model to 1.3b

# Connect components in the query pipeline
query_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
# Connect retriever's documents to the passthrough component
query_pipeline.connect("retriever.documents", "doc_passthrough.documents")
# Connect passthrough component's documents to prompt_builder
query_pipeline.connect("doc_passthrough.documents", "prompt_builder.documents")
# Connect prompt_builder's output to the new converter
query_pipeline.connect("prompt_builder.prompt", "string_to_chat_message_converter.prompt")
# Connect converter's output (list of ChatMessage) to LLM's messages input
query_pipeline.connect("string_to_chat_message_converter.messages", "llm.messages")
print("Haystack RAG query pipeline set up.")

# --- Step 5: Interact with the Chatbot ---
print("\n--- Q&A Chatbot Ready (using Haystack and DeepSeek via Ollama) ---")
print("Ask questions about the Amazon rainforest (or your document content).")
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
        # Run the query pipeline
        # Now, only the outputs of the terminal component (llm) are directly accessible
        response = query_pipeline.run(
            {
                "query_embedder": {"text": query},
                "prompt_builder": {"question": query},
            }
        )

        # Extract the answer from the LLM. The key is the component name.
        answer = response["llm"]["replies"][0]

        # In this Haystack version, intermediate outputs (like retrieved_documents from doc_passthrough)
        # are NOT directly accessible from the top-level 'response' dictionary if 'add_outputs'
        # or 'debug_outputs' are not supported.
        # Therefore, the 'Source Documents Used' section is commented out to avoid errors.
        # If you need to see the retrieved documents, you might need to upgrade Haystack
        # or use a different debugging/logging mechanism within the components themselves.
        # retrieved_documents = response["doc_passthrough"]["documents"]


        print(f"\nChatbot Answer: {answer}")

        # if retrieved_documents:
        #     print("\nSource Documents Used:")
        #     for i, doc in enumerate(retrieved_documents):
        #         print(f"  Chunk {i+1}: {doc.content[:150]}...")
        #         if doc.meta:
        #             print(f"    Metadata: {doc.meta}")
        # else:
        #     print("No specific source documents were identified for this query.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure Ollama is installed and running, and the DeepSeek model is pulled.")
        print("Also check the model name in OllamaChatGenerator matches your pulled model.")
