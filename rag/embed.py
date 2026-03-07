from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], prefix: str = "passage: ") -> list[list[float]]:
    """Embed a list of texts. Use prefix='query: ' for search queries."""
    model = _get_model()
    prefixed = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single search query."""
    return embed_texts([text], prefix="query: ")[0]
