# run_chatbot.py

from haystack import Pipeline, component, Document
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

# --- Custom Components ---
@component
class StringToChatMessageConverter:
    @component.output_types(messages=list[ChatMessage])
    def run(self, prompt: str):
        return {"messages": [ChatMessage.from_user(prompt)]}

@component
class DocumentPassthrough:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        return {"documents": documents}

# Load existing Milvus store (don't drop)
document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus_data.db"},
    drop_old=False,
    collection_name="docling_crawled_docs"
)
print("Connected to existing MilvusDocumentStore.")

# Create RAG Pipeline
query_pipeline = Pipeline()
query_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
query_pipeline.add_component("retriever", MilvusEmbeddingRetriever(document_store=document_store))
query_pipeline.add_component("passthrough", DocumentPassthrough())
query_pipeline.add_component("prompt_builder", PromptBuilder(template="""
Given the following context, answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{% for document in documents %}
    {{ document.content }}
    Source: {{ document.meta.source }}
{% endfor %}

Question: {{ question }}
Answer:
""", required_variables=["documents", "question"]))
query_pipeline.add_component("converter", StringToChatMessageConverter())
query_pipeline.add_component("llm", OllamaChatGenerator(model="qwen3:0.6b", timeout=300))

# Connect components
query_pipeline.connect("embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "passthrough.documents")
query_pipeline.connect("passthrough.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder.prompt", "converter.prompt")
query_pipeline.connect("converter.messages", "llm.messages")

# Chat loop
print("\n--- Q&A Chatbot Ready ---")
# print("Ask your question or type 'exit' to quit.")

# while True:
#     question = input("\nYour question: ").strip()
#     if question.lower() == "exit":
#         print("Goodbye!")
#         break
#     if not question:
#         continue

#     try:
#         result = query_pipeline.run({
#             "embedder": {"text": question},
#             "prompt_builder": {"question": question}
#         })
#         answer = result["llm"]["replies"][0].text
#         print(f"\nChatbot Answer: {answer}")
#     except Exception as e:
#         print(f"Error during Q&A: {e}")
#         print("Check if Ollama is running and model is available.")

result = query_pipeline.run({
            "embedder": {"text": "How to create a linux instance"},
            "prompt_builder": {"question": "How to create a linux instance"}
        })
answer = result["llm"]["replies"][0].text
print(f"\nChatbot Answer: {answer}")
print('-----------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------')
result = query_pipeline.run({
            "embedder": {"text": "How to create a windows instance"},
            "prompt_builder": {"question": "How to create a windows instance"}
        })
answer = result["llm"]["replies"][0].text
print(f"\nChatbot Answer: {answer}")
print('-----------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------')