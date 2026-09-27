"""Memory-efficient sparse BM25 based on SciPy CSR matrices.

Online scoring uses an in-memory inverted index (a CSC copy of the same
matrix): a query touches only the posting lists of its own terms instead of
all 33.6M non-zeros of the document-term matrix.
"""

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
    # Inverted index: column j holds the posting list (documents, BM25 weights)
    # of term j. Built from `matrix` on first use; never serialized.
    postings: sp.csc_matrix | None = None

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
        self.postings = None  # rebuilt lazily for the new matrix
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
        scores = self.score(query)
        if allowed_mask is not None:
            scores = np.where(allowed_mask, scores, -np.inf)
        valid_count = int(np.isfinite(scores).sum())
        k = min(top_k, valid_count)
        if k == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        candidate = np.argpartition(scores, -k)[-k:]
        order = candidate[np.argsort(scores[candidate])[::-1]]
        return order, scores[order]

    def build_postings(self) -> sp.csc_matrix:
        """Build (once) the in-memory inverted index used by `score`."""
        if self.matrix is None:
            raise RuntimeError("Call fit() or load() before build_postings()")
        if self.postings is None:
            postings = self.matrix.tocsc()
            postings.sort_indices()
            self.postings = postings
        return self.postings

    def score(self, query: str) -> np.ndarray:
        """Compute one corpus score vector for reuse by global/local channels.

        Query term weights are binary, so a document score is the sum of its
        BM25 weights over the distinct query terms. Posting lists are added in
        ascending term id with float32 accumulation — the same order and
        precision SciPy uses for `matrix @ query.T` — so the result is
        bit-identical to the full sparse product (`score_matmul`), only the
        work is proportional to the query's postings, not to the whole matrix.
        """
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Call fit() or load() before score()")
        postings = self.build_postings()
        terms = np.unique(self.vectorizer.transform([query]).indices)
        scores = np.zeros(postings.shape[0], dtype=np.float32)
        for term in terms:
            start, end = postings.indptr[term], postings.indptr[term + 1]
            scores[postings.indices[start:end]] += postings.data[start:end]
        return scores

    def score_matmul(self, query: str) -> np.ndarray:
        """Reference scorer: full sparse product over the document-term matrix."""
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Call fit() or load() before score()")
        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz:
            query_vector.data[:] = 1.0
        return np.asarray((self.matrix @ query_vector.T).toarray(), dtype=np.float32).ravel()

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
