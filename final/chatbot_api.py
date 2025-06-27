from flask import Flask, request, jsonify
app = Flask(__name__)
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

@app.route('/')
def home():
    return "Welcome to the Chat Bot"

@app.route('/ask-question', methods=['POST'])
def ask_question():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    payload = request.get_json()

    if not payload or 'question' not in payload:
        return jsonify({"error": "question is required"}), 400

    question = payload.get("question")

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


    result = query_pipeline.run({
            "embedder": {"text": "How to create a linux instance"},
            "prompt_builder": {"question": "How to create a linux instance"}
        })
    # answer = result["llm"]["replies"][0].text
    
    return jsonify(result["llm"]["replies"][0]), 201

if __name__ == '__main__':
    app.run(debug=True)