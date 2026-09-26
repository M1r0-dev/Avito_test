"""Memory-efficient sparse BM25 based on SciPy CSR matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .text import tokenize


@dataclass
class SparseBM25:
    """A serializable BM25 index with optional row-level pre-filtering."""

    k1: float = 1.5
    b: float = 0.75
    min_df: int = 2
    max_features: int | None = 350_000
    vectorizer: CountVectorizer | None = None
    matrix: sp.csr_matrix | None = None

    def fit(self, documents: list[str]) -> "SparseBM25":
        self.vectorizer = CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            min_df=self.min_df,
            max_features=self.max_features,
            dtype=np.float32,
        )
        counts = self.vectorizer.fit_transform(documents).tocsr()
        n_documents = counts.shape[0]
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((n_documents - document_frequency + 0.5) / (document_frequency + 0.5))

        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = max(float(lengths.mean()), 1.0)
        row_ids = np.repeat(np.arange(n_documents), np.diff(counts.indptr))
        normalizer = self.k1 * (1.0 - self.b + self.b * lengths[row_ids] / average_length)
        counts.data = counts.data * (self.k1 + 1.0) / (counts.data + normalizer)
        counts.data *= idf[counts.indices]
        self.matrix = counts.astype(np.float32)
        return self

    def search(
        self,
        query: str,
        *,
        top_k: int = 250,
        allowed_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return row indices and BM25 scores in descending order."""
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Call fit() or load() before search()")
        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz:
            query_vector.data[:] = 1.0
        scores = np.asarray((self.matrix @ query_vector.T).toarray()).ravel()
        if allowed_mask is not None:
            scores = np.where(allowed_mask, scores, -np.inf)
        valid_count = int(np.isfinite(scores).sum())
        k = min(top_k, valid_count)
        if k == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        candidate = np.argpartition(scores, -k)[-k:]
        order = candidate[np.argsort(scores[candidate])[::-1]]
        return order, scores[order]

    def score_many(self, queries: list[str]) -> np.ndarray:
        """Score a query batch in one sparse multiplication (docs × queries)."""
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Call fit() or load() before search()")
        query_matrix = self.vectorizer.transform(queries)
        if query_matrix.nnz:
            query_matrix.data[:] = 1.0
        return np.asarray((self.matrix @ query_matrix.T).toarray(), dtype=np.float32)

    def save(self, directory: str | Path) -> None:
        """Persist vocabulary/config separately from the sparse matrix."""
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Cannot save an unfitted index")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "k1": self.k1,
                "b": self.b,
                "min_df": self.min_df,
                "max_features": self.max_features,
                "vectorizer": self.vectorizer,
            },
            directory / "bm25.joblib",
        )
        sp.save_npz(directory / "bm25_matrix.npz", self.matrix)

    @classmethod
    def load(cls, directory: str | Path) -> "SparseBM25":
        directory = Path(directory)
        state = joblib.load(directory / "bm25.joblib")
        obj = cls(**{k: state[k] for k in ("k1", "b", "min_df", "max_features")})
        obj.vectorizer = state["vectorizer"]
        obj.matrix = sp.load_npz(directory / "bm25_matrix.npz").tocsr()
        return obj
