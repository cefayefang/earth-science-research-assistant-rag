from functools import lru_cache
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path
from ..core.config import get_settings, ROOT
from ..core.schemas import NormalizedDataset, Chunk


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    cfg = get_settings()
    model_name = cfg["embeddings"]["model_name"]
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    cfg = get_settings()
    chroma_dir = ROOT / cfg["paths"]["chroma_dir"]
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_dataset_collection() -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="datasets",
        metadata={"hnsw:space": "cosine"},
    )


def get_chunk_collection() -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="chunks",
        metadata={"hnsw:space": "cosine"},
    )


def embed_datasets(datasets: list[NormalizedDataset], batch_size: int = 64) -> None:
    model = get_embedding_model()
    collection = get_dataset_collection()

    existing = set(collection.get()["ids"])
    new_datasets = [d for d in datasets if d.dataset_id not in existing]

    if not new_datasets:
        print("Dataset collection already up to date.")
        return

    texts = [d.retrieval_text for d in new_datasets]
    ids = [d.dataset_id for d in new_datasets]
    metadatas = [
        {
            "source": d.source,
            "title": d.display_name[:500],
            "doi": d.doi or "",
            "spatial_info": d.spatial_info or "",
            "temporal_info": d.temporal_info or "",
            "keywords": ",".join(d.keywords[:20]),
            "variables": ",".join(d.variables[:20]),
        }
        for d in new_datasets
    ]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
        collection.add(ids=batch_ids, embeddings=embeddings, metadatas=batch_meta, documents=batch_texts)
        print(f"  Embedded datasets {i + 1}–{min(i + batch_size, len(texts))}/{len(texts)}")

    print(f"Dataset collection: {collection.count()} total records")


def embed_chunks(chunks: list[Chunk], batch_size: int = 64) -> None:
    model = get_embedding_model()
    collection = get_chunk_collection()

    existing = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c.chunk_id not in existing]

    if not new_chunks:
        print("Chunk collection already up to date.")
        return

    texts = [c.text for c in new_chunks]
    ids = [c.chunk_id for c in new_chunks]
    metadatas = [
        {
            "local_id": c.local_id,
            "openalex_id": c.openalex_id or "",
            "filename": c.filename,
            "section_guess": c.section_guess or "",
            "page_start": c.page_range[0],
            "page_end": c.page_range[1],
        }
        for c in new_chunks
    ]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
        collection.add(ids=batch_ids, embeddings=embeddings, metadatas=batch_meta, documents=batch_texts)
        print(f"  Embedded chunks {i + 1}–{min(i + batch_size, len(texts))}/{len(texts)}")

    print(f"Chunk collection: {collection.count()} total records")


def query_embedding(text: str) -> list[float]:
    model = get_embedding_model()
    return model.encode([text], normalize_embeddings=True)[0].tolist()
