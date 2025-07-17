from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from _core.config import config


class BaseEmbeddingManager(ABC):
    """Abstract base class for embedding managers."""

    @abstractmethod
    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Convert text(s) into embeddings."""
        pass


class SentenceTransformerEmbeddingManager(BaseEmbeddingManager):
    """Manages text embedding operations using Sentence Transformers models."""

    def __init__(self):
        """Initialize the embedding manager."""
        self._model: SentenceTransformer = SentenceTransformer(
            model_name_or_path=config["sentence_transformers"]["model_path"]
        )

    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for input text(s)."""
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = self._model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e


st_encoder = SentenceTransformerEmbeddingManager()
