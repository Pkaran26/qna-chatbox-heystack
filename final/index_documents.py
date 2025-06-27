# index_documents.py

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from milvus_haystack import MilvusDocumentStore
from docling.document_converter import DocumentConverter
import os

# Load URLs
crawled_document_path = "./crawled_documents.txt"
urls_to_crawl = []
if os.path.exists(crawled_document_path):
    with open(crawled_document_path, "r", encoding="utf-8") as f:
        urls_to_crawl = [line.strip() for line in f if line.strip()]
    print(f"Found {len(urls_to_crawl)} URLs in {crawled_document_path}")
else:
    print(f"File not found: {crawled_document_path}")
    exit(1)

# Initialize Milvus Document Store
document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus_data.db"},
    drop_old=True,
    collection_name="docling_crawled_docs"
)
print("MilvusDocumentStore initialized.")

# Convert URLs to Haystack Documents using Docling
docling_converter = DocumentConverter()
crawled_haystack_documents = []

for url in urls_to_crawl:
    try:
        print(f"Processing URL: {url}")
        result = docling_converter.convert(url)
        if result and result.document:
            content = result.document.export_to_markdown()
            haystack_doc = Document(content=content, meta={"source": url, "processed_by": "docling"})
            crawled_haystack_documents.append(haystack_doc)
            print(f"Added document from {url}")
        else:
            print(f"Empty result for {url}")
    except Exception as e:
        print(f"Error processing {url}: {e}")

# Split documents into passages
splitter = DocumentSplitter(split_by="passage", split_length=1000, split_overlap=200)
split_documents = splitter.run(documents=crawled_haystack_documents)["documents"]
print(f"Split into {len(split_documents)} document chunks.")

# Build and run indexing pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
indexing_pipeline.connect("embedder.documents", "writer.documents")

if split_documents:
    print("Running indexing pipeline...")
    indexing_pipeline.run({"embedder": {"documents": split_documents}})
    print(f"Indexing complete. Total documents in Milvus: {document_store.count_documents()}")
else:
    print("No documents to index.")
