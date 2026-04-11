import sys, os
from pathlib import Path
import numpy as np

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAU2_SRC = PROJECT_ROOT / "benchmarks" / "tau2_bench" / "repo" / "src"
if str(TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(TAU2_SRC))

from tau2.domains.banking_knowledge.environment import get_knowledge_base
from tau2.knowledge.document_preprocessors.chunked_embedding_indexer import (
    ChunkedEmbeddingIndexer,
)

from tau2.knowledge.input_preprocessors.embedding_encoder import EmbeddingEncoder


load_dotenv(PROJECT_ROOT / ".env")

kb = get_knowledge_base()
docs = [
    {
        "id": doc.id,
        "title": doc.title,
        "text": doc.content,
    }
    for doc in kb.get_all_documents()
]

target = max(docs, key=lambda d: len(d["text"]))

print(f"Total documents: {len(docs)}")
print(f"Found document with {len(target['text'])} characters")
print(f"Document ID: {target['id']}")
print(f"Document title: {target['title']}")

indexer = ChunkedEmbeddingIndexer(
    use_cache=True,
    chunk_size=900,
    chunk_overlap=150,
    min_chunk_chars=120,
    include_title=True,
)

chunk_docs = indexer._build_chunk_docs([target])

top_k = 3

print("=" * 120)
print("CHUNKING INSPECTION")
print("=" * 120)
print(f"DOC_ID: {target['id']}")
print(f"TITLE: {target['title']}")
print(f"ORIGINAL_CHARS: {len(target['text'])}")
print(f"CHUNK_SIZE: {indexer.chunk_size}")
print(f"CHUNK_OVERLAP: {indexer.chunk_overlap}")
print(f"MIN_CHUNK_CHARS: {indexer.min_chunk_chars}")
print(f"INCLUDE_TITLE: {indexer.include_title}")
print(f"NUM_CHUNKS: {len(chunk_docs)}")
print()

for i, chunk in enumerate(chunk_docs):
    text = chunk["text"]
    print("-" * 120)
    print(f"CHUNK_INDEX: {i}")
    print(f"CHUNK_ID: {chunk['id']}")
    print(f"CHUNK_CHARS: {len(text)}")
    print("CHUNK_TEXT_START")
    print(text)
    print("CHUNK_TEXT_END")
    print()

# Testing embedder

model = os.environ["EMBEDDING_MODEL"]

doc_embedder = indexer._get_embedder()
query_embedder = EmbeddingEncoder(
    embedder_type="openai",
    embedder_params={"model": model},
)._get_embedder()


chunk_texts = [chunk["text"] for chunk in chunk_docs]
query = "When can an agent unlock a debit card PIN after fraud review?"

doc_embeddings = doc_embedder.embed(chunk_texts)
query_embedding = query_embedder.embed([query])[0]

def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("=" * 120)
print("EMBEDDING INSPECTION")
print("=" * 120)
print(f"MODEL: {model}")
print(f"DOC_ID: {target['id']}")
print(f"TITLE: {target['title']}")
print(f"NUM_CHUNKS: {len(chunk_docs)}")
print(f"TOP K: {top_k}")
print(f"USER_QUERY: {query}")
print(f"DOC_EMBEDDER_QUERY_INSTRUCTION: {repr(getattr(doc_embedder, 'query_instruction', None))}")
print(f"QUERY_EMBEDDER_QUERY_INSTRUCTION: {repr(getattr(query_embedder, 'query_instruction', None))}")
print(f"DOC_EMBEDDINGS_SHAPE: {doc_embeddings.shape}")
print(f"QUERY_EMBEDDING_DIM: {query_embedding.shape[0]}")
print()

scores = [
    (i, cosine(query_embedding, doc_embeddings[i]))
    for i in range(len(chunk_docs))
]
scores.sort(key=lambda x: x[1], reverse=True)

for rank, (i, score) in enumerate(scores[:top_k], start=1):
    chunk = chunk_docs[i]
    print("-" * 120)
    print(f"RANK: {rank}")
    print(f"CHUNK_INDEX: {i}")
    print(f"CHUNK_ID: {chunk['id']}")
    print(f"CHUNK_CHARS: {len(chunk['text'])}")
    print(f"COSINE(query, chunk): {score:.6f}")
    print("CHUNK_PREVIEW_START")
    print(chunk["text"][:500])
    print("CHUNK_PREVIEW_END")
    print()

formatted_query = query_embedder._format_text(query)
print("=" * 120)
print("FORMATTED QUERY SENT TO EMBEDDER")
print("=" * 120)
print(formatted_query)