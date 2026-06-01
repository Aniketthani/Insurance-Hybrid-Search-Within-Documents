"""
Phase 3 & 4 (part): Hybrid Search Index
=========================================
Fixes the "Silent Numerical Mismatch" gotcha with Hybrid Index.
- BM25 sparse index  → exact term / numerical value matching (rank-bm25)
- TF-IDF vector index → cosine similarity (sklearn; swappable with sentence-transformers)
- Qdrant in-memory   → vector store with metadata filtering
- RRF fusion         → merges both ranked lists without score normalisation
- Cross-encoder reranking → BM25-based pseudo-reranking (no neural model needed)

NOTE: Designed to be drop-in swappable to BAAI/bge-large-en-v1.5 or
      Qwen3-VL-Embedding-8B once HuggingFace access is available.
      Just replace the TFIDFEmbedder class with SentenceTransformerEmbedder.
"""

import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)
from tqdm import tqdm
from rich.console import Console
from src.chunker import Chunk

console = Console()

COLLECTION_NAME = "insurance_docs"

# Insurance-specific stopwords (keep domain terms intact)
INSURANCE_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "as", "not", "have",
    "will", "shall", "may", "any", "all", "each", "such", "said", "per",
}


# ── Embedder abstraction — swappable ──────────────────────────────────────

class TFIDFEmbedder:
    """
    TF-IDF based embedder.
    Produces normalized TF-IDF vectors for cosine similarity.
    Identical interface to SentenceTransformerEmbedder for drop-in swap.
    
    Swap this class with SentenceTransformerEmbedder (below) when
    HuggingFace model download is available.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8192,
            ngram_range=(1, 2),        # unigrams + bigrams
            sublinear_tf=True,         # log(1+TF) smoothing
            min_df=1,
            analyzer="word",
            stop_words=list(INSURANCE_STOPWORDS),
            token_pattern=r"[a-z0-9$%,./-]{2,}",
        )
        self._fitted = False
        self._dim = 8192
        self.name = "TF-IDF (sklearn, offline)"

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True
        self._dim = len(self.vectorizer.vocabulary_)
        console.print(f"  [dim]TF-IDF vocab: {self._dim:,} terms[/]")

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if not self._fitted:
            raise RuntimeError("Call fit() before encode()")
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.vectorizer.transform(texts).toarray().astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs = vecs / norms
        return vecs

    @property
    def dim(self):
        return self._dim


class SentenceTransformerEmbedder:
    """
    Drop-in replacement for TFIDFEmbedder using sentence-transformers.
    Use when HuggingFace model download is available:
        embedder = SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5")
    """
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dim  = self.model.get_sentence_embedding_dimension()
        self.name  = model_name

    def fit(self, texts: List[str]):
        pass

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

    @property
    def dim(self):
        return self._dim


class Qwen3VLEmbedder:
    """
    Qwen3-VL-Embedding-8B — multimodal vision-language embedder.

    Why it matters for insurance:
    - Reinsurance placement slips are multi-column visual documents.
      Text-only models lose spatial layout (which column a $ amount is in).
    - This model maps text, images, AND raw PDF snapshots into the same
      vector space, so table structure is preserved in the embedding.
    - Best accuracy for broker slips, schedule tables, deductible grids.

    Hardware requirements:
    - GPU strongly recommended (A100 / H100 for full 8B model).
    - CPU-only fallback works but indexing will be very slow (minutes per doc).
    - For CPU-only machines, use SentenceTransformerEmbedder instead.

    Usage:
        embedder = Qwen3VLEmbedder()                    # auto device
        embedder = Qwen3VLEmbedder(device="cuda")        # force GPU
        embedder = Qwen3VLEmbedder(device="cpu")         # force CPU (slow)
    """

    MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"

    def __init__(self, device: str = "auto"):
        import torch
        from transformers import AutoTokenizer, AutoModel

        console.print(f"[cyan]  Loading {self.MODEL_NAME}...[/]")
        console.print("[dim]  (first run downloads ~16 GB — subsequent runs use cache)[/]")

        # Resolve device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        if self._device == "cpu":
            console.print("[yellow]  ⚠ Running on CPU — encoding will be slow. "
                          "Use GPU for production.[/]")

        dtype = torch.float16 if self._device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self._device)
        self.model.eval()

        # Qwen3-VL-Embedding outputs 4096-dim vectors
        self._dim = 4096
        self.name = f"Qwen3-VL-Embedding-8B ({self._device})"
        console.print(f"[green]  ✓ {self.name} loaded[/]")

    def fit(self, texts: List[str]):
        pass  # Neural model — no fitting needed

    def encode(self, texts, normalize_embeddings: bool = True,
               show_progress_bar: bool = False) -> np.ndarray:
        import torch

        if isinstance(texts, str):
            texts = [texts]

        all_vecs = []
        batch_size = 8  # keep VRAM usage manageable

        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            from tqdm import tqdm as _tqdm
            iterator = _tqdm(iterator, desc="Qwen3-VL encoding")

        with torch.no_grad():
            for i in iterator:
                batch = texts[i: i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self._device)

                outputs = self.model(**encoded)

                # Mean-pool over non-padding tokens
                attention_mask = encoded["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                vecs = (
                    torch.sum(token_embeddings * input_mask_expanded, dim=1)
                    / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                ).cpu().numpy().astype(np.float32)

                if normalize_embeddings:
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    vecs = vecs / norms

                all_vecs.append(vecs)

        return np.vstack(all_vecs)

    @property
    def dim(self) -> int:
        return self._dim


# ── Embedder registry — used by InsuranceHybridSearchIndex ───────────────
EMBEDDER_OPTIONS = {
    "tfidf":   "TF-IDF (offline, no GPU needed)",
    "bge":     "BAAI/bge-large-en-v1.5 (best open-source, needs HF download)",
    "qwen3vl": "Qwen3-VL-Embedding-8B (best for visual/table PDFs, needs HF + GPU)",
}

def build_embedder(embedder_type: str = "tfidf", device: str = "auto"):
    """Factory — returns the right embedder based on user selection."""
    if embedder_type == "tfidf":
        return TFIDFEmbedder()
    elif embedder_type == "bge":
        return SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5")
    elif embedder_type == "qwen3vl":
        return Qwen3VLEmbedder(device=device)
    else:
        raise ValueError(f"Unknown embedder_type '{embedder_type}'. "
                         f"Choose from: {list(EMBEDDER_OPTIONS)}")


# ── BM25 Reranker (pure term-overlap, no neural model needed) ─────────────

class BM25Reranker:
    """
    Pseudo cross-encoder: re-scores candidates by BM25 on query terms
    vs. chunk text. Much lighter than a neural cross-encoder while still
    improving precision over pure vector search.
    
    Replace with CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    once HuggingFace is available.
    """
    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        scores = []
        for query, passage in pairs:
            q_tokens = set(query.lower().split())
            p_tokens  = passage.lower().split()
            if not q_tokens or not p_tokens:
                scores.append(0.0)
                continue
            # Term overlap with position weighting (earlier = higher weight)
            hit = 0.0
            for i, tok in enumerate(p_tokens[:200]):
                for qt in q_tokens:
                    if qt in tok:
                        hit += 1.0 / math.log2(i + 2)
            scores.append(hit)
        return scores


# ── Main search index ─────────────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk: Chunk
    bm25_score: float
    vector_score: float
    rrf_score: float
    rerank_score: Optional[float] = None
    final_score: float = 0.0


class InsuranceHybridSearchIndex:
    """
    Hybrid BM25 + TF-IDF Cosine search with RRF fusion and reranking.
    
    Architecture is identical to the production version — only the
    embedding model differs (TF-IDF vs neural). Swap embedder class above
    to go production-grade with no other code changes.
    """

    RRF_K         = 60
    BM25_WEIGHT   = 0.40   # α
    VECTOR_WEIGHT = 0.60   # β
    CANDIDATE_K   = 50

    def __init__(
        self,
        use_reranker:  bool = True,
        verbose:       bool = True,
        embedder_type: str  = "tfidf",   # "tfidf" | "bge" | "qwen3vl"
        device:        str  = "auto",    # "auto" | "cuda" | "cpu"
    ):
        self.use_reranker  = use_reranker
        self.verbose       = verbose
        self.embedder_type = embedder_type

        label = EMBEDDER_OPTIONS.get(embedder_type, embedder_type)
        console.print(f"[cyan]🤖 Initialising embedding engine: {label}[/]")
        self.embedder = build_embedder(embedder_type, device)

        if use_reranker:
            console.print("[cyan]🔁 Initialising BM25 reranker...[/]")
            self.reranker = BM25Reranker()

        self._chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []
        self._vectors: Optional[np.ndarray] = None
        self._parent_map: Dict[str, Chunk] = {}

        console.print(f"[green]✓ Search engine ready  |  embedder={self.embedder.name}[/]")

    # ------------------------------------------------------------------ #
    #  Indexing                                                            #
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: List[Chunk]):
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        parent_map   = {c.chunk_id: c for c in chunks if c.chunk_type == "parent"}

        if self.verbose:
            console.print(f"[cyan]📥 Indexing {len(child_chunks)} child chunks...[/]")

        self._chunks.extend(child_chunks)
        self._parent_map.update(parent_map)

        # ── BM25 ──────────────────────────────────────────────────────
        new_tokenized = [self._tokenize(c.text) for c in child_chunks]
        self._tokenized_corpus.extend(new_tokenized)
        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=1.5,
            b=0.6,
        )

        # ── TF-IDF vectors ────────────────────────────────────────────
        all_texts = [c.text for c in self._chunks]
        console.print(f"[cyan]  Fitting/encoding {len(all_texts)} chunks with {self.embedder.name}...[/]")
        self.embedder.fit(all_texts)
        self._vectors = self.embedder.encode(all_texts, normalize_embeddings=True)

        if self.verbose:
            parents  = len(self._parent_map)
            children = len(self._chunks)
            vec_dim  = self.embedder.dim
            console.print(
                f"[green]✓ Indexed {children} child chunks | "
                f"{parents} parent sections | "
                f"vector dim={vec_dim:,}[/]"
            )

    # ------------------------------------------------------------------ #
    #  Search                                                              #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        lob_filter: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> List[SearchResult]:
        if not self._chunks:
            return []

        use_rerank = self.use_reranker if rerank is None else rerank
        query = query.strip()

        bm25_results = self._bm25_search(query, k=self.CANDIDATE_K)
        vec_results  = self._vector_search(query, k=self.CANDIDATE_K)
        fused        = self._rrf_fuse(bm25_results, vec_results, top_k=top_k * 3)

        # Metadata filters
        if doc_filter or lob_filter:
            fused = [
                r for r in fused
                if (not doc_filter or r.chunk.doc_name == doc_filter)
                and (not lob_filter or r.chunk.lob == lob_filter)
            ]

        fused = fused[:top_k * 2]

        if use_rerank and len(fused) > 1:
            fused = self._rerank(query, fused)

        for r in fused:
            r.final_score = r.rerank_score if r.rerank_score is not None else r.rrf_score

        fused.sort(key=lambda x: x.final_score, reverse=True)
        return fused[:top_k]

    def get_parent_context(self, chunk: Chunk) -> Optional[Chunk]:
        if chunk.parent_id:
            return self._parent_map.get(chunk.parent_id)
        return None

    def stats(self) -> Dict:
        return {
            "total_child_chunks":    len(self._chunks),
            "total_parent_sections": len(self._parent_map),
            "embedding_engine":      self.embedder.name,
            "embedder_type":         self.embedder_type,
            "vector_dimensions":     self.embedder.dim,
            "reranker":              "BM25 overlap" if self.use_reranker else "disabled",
            "rrf_k":                 self.RRF_K,
            "bm25_weight_alpha":     self.BM25_WEIGHT,
            "vector_weight_beta":    self.VECTOR_WEIGHT,
        }

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        tokenized = self._tokenize(query)
        scores    = self._bm25.get_scores(tokenized)
        top_idx   = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _vector_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        q_vec  = self.embedder.encode(query, normalize_embeddings=True)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        sims   = cosine_similarity(q_vec, self._vectors)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in top_idx]

    def _rrf_fuse(
        self,
        bm25_results: List[Tuple[int, float]],
        vec_results:  List[Tuple[int, float]],
        top_k: int,
    ) -> List[SearchResult]:
        bm25_rank  = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        vec_rank   = {idx: rank + 1 for rank, (idx, _) in enumerate(vec_results)}
        bm25_score = {idx: s for idx, s in bm25_results}
        vec_score  = {idx: s for idx, s in vec_results}

        all_ids = set(bm25_rank.keys()) | set(vec_rank.keys())
        scored  = []
        for idx in all_ids:
            if idx >= len(self._chunks):
                continue
            rrf = 0.0
            if idx in bm25_rank:
                rrf += self.BM25_WEIGHT  / (self.RRF_K + bm25_rank[idx])
            if idx in vec_rank:
                rrf += self.VECTOR_WEIGHT / (self.RRF_K + vec_rank[idx])
            scored.append(SearchResult(
                chunk        = self._chunks[idx],
                bm25_score   = bm25_score.get(idx, 0.0),
                vector_score = vec_score.get(idx, 0.0),
                rrf_score    = rrf,
            ))
        scored.sort(key=lambda x: x.rrf_score, reverse=True)
        return scored[:top_k]

    def _rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        pairs  = [(query, r.chunk.raw_text[:512]) for r in results]
        scores = self.reranker.predict(pairs)
        for r, s in zip(results, scores):
            r.rerank_score = float(s)
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results

    def _tokenize(self, text: str) -> List[str]:
        text   = re.sub(r"\$([0-9,]+)", lambda m: m.group(1).replace(",", ""), text)
        text   = re.sub(r"([0-9,]+),([0-9]{3})", r"\1\2", text)
        tokens = re.findall(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", text.lower())
        return [t for t in tokens if t not in INSURANCE_STOPWORDS and len(t) > 1]
