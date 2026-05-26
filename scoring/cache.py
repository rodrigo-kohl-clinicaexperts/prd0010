"""
Caches persistentes para o pipeline de scoring.

Dois tipos:

1. EmbeddingCache — para encoders que produzem 1 vetor por texto
   (BGE dense+sparse, E5 dense).
       /hashes        (N,)       vlen utf-8     sha256 hex de cada texto
       /dense         (N, D)     float32        vetor denso normalizado
       /sparse_json   (N,)       vlen utf-8     JSON {token_id: peso}  (opt)

2. TokenEmbeddingCache — para encoders que produzem uma matriz token-level
   (T_i, D) por texto, com T_i variável (BGE colbert_vecs, etc).
       /hashes        (N,)         vlen utf-8     sha256 hex
       /offsets       (N+1,)       int64          offsets em /tokens_flat
       /tokens_flat   (sum_T, D)   float32        embeddings concatenados
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import h5py
import numpy as np


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path | str, has_sparse: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.has_sparse = has_sparse
        self._f: h5py.File | None = h5py.File(self.path, "a")
        if "hashes" in self._f:
            existing = self._f["hashes"].asstr()[:]
            self._index: dict[str, int] = {h: i for i, h in enumerate(existing)}
        else:
            self._index = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __len__(self):
        return len(self._index)

    # ------------------------------------------------------------------

    def _ensure_datasets(self, dim: int):
        str_dtype = h5py.string_dtype(encoding="utf-8")
        if "dense" not in self._f:
            self._f.create_dataset(
                "dense",
                shape=(0, dim),
                maxshape=(None, dim),
                dtype="float32",
                chunks=True,
                compression="lzf",
            )
        if "hashes" not in self._f:
            self._f.create_dataset(
                "hashes",
                shape=(0,),
                maxshape=(None,),
                dtype=str_dtype,
                chunks=True,
            )
        if self.has_sparse and "sparse_json" not in self._f:
            self._f.create_dataset(
                "sparse_json",
                shape=(0,),
                maxshape=(None,),
                dtype=str_dtype,
                chunks=True,
                compression="lzf",
            )

    def _append(
        self,
        new_hashes: list[str],
        new_dense: np.ndarray,
        new_sparse_json: list[str] | None,
    ):
        n_old = self._f["dense"].shape[0]
        n_new = len(new_hashes)

        self._f["dense"].resize(n_old + n_new, axis=0)
        self._f["dense"][n_old:] = new_dense.astype("float32", copy=False)

        self._f["hashes"].resize(n_old + n_new, axis=0)
        self._f["hashes"][n_old:] = np.asarray(new_hashes, dtype=object)

        if self.has_sparse:
            self._f["sparse_json"].resize(n_old + n_new, axis=0)
            self._f["sparse_json"][n_old:] = np.asarray(new_sparse_json, dtype=object)

        for i, h in enumerate(new_hashes):
            self._index[h] = n_old + i
        self._f.flush()

    # ------------------------------------------------------------------

    def get_or_encode(
        self,
        texts: list[str],
        encode_fn: Callable[[list[str]], dict],
        *,
        label: str = "",
        append_every: int = 256,
    ) -> dict:
        """
        Para cada texto, retorna o embedding (do cache ou recém-codificado).

        encode_fn(faltantes) deve devolver um dict com:
            "dense":  np.ndarray (M, D)
            "sparse": list[dict[int, float]]   (apenas se has_sparse=True)

        append_every: tamanho do chunk de append. Os faltantes são divididos
        em blocos de até `append_every` textos. A cada bloco: encoda → grava
        no .h5 → flush. Se o processo morrer no meio, o que já foi gravado
        sobrevive. Valores maiores = menos I/O mas mais perda em caso de
        crash; valores menores = mais seguro mas mais I/O.

        Retorno (na ordem de `texts`):
            "dense":  np.ndarray (len(texts), D)
            "sparse": list[dict[int, float]] | None
        """
        if not texts:
            return {"dense": np.zeros((0, 0), dtype="float32"),
                    "sparse": [] if self.has_sparse else None}
        if append_every < 1:
            raise ValueError(f"append_every deve ser >= 1, recebi {append_every}")

        hashes = [_hash(t) for t in texts]
        missing_pos = [i for i, h in enumerate(hashes) if h not in self._index]
        prefix = f"  cache[{label}]" if label else "  cache"

        if missing_pos:
            hit = len(texts) - len(missing_pos)
            total_missing = len(missing_pos)
            print(f"{prefix}: {hit}/{len(texts)} hits, codificando "
                  f"{total_missing} novo(s) em chunks de {append_every}...")

            for chunk_start in range(0, total_missing, append_every):
                chunk_end = min(chunk_start + append_every, total_missing)
                chunk_positions = missing_pos[chunk_start:chunk_end]
                chunk_texts = [texts[i] for i in chunk_positions]
                chunk_hashes = [hashes[i] for i in chunk_positions]

                out = encode_fn(chunk_texts)
                dense_new = np.asarray(out["dense"])
                self._ensure_datasets(int(dense_new.shape[1]))

                sparse_new_json: list[str] | None = None
                if self.has_sparse:
                    sparse_new_json = [
                        json.dumps({int(k): float(v) for k, v in d.items()})
                        for d in out["sparse"]
                    ]

                self._append(chunk_hashes, dense_new, sparse_new_json)
                print(f"{prefix}: flush {chunk_end}/{total_missing}")
        else:
            print(f"{prefix}: 100% hit ({len(texts)} texto(s)).")

        # h5py fancy indexing exige índices ordenados e únicos; usamos
        # np.unique + inverse para reconstruir a ordem original.
        indices = np.fromiter(
            (self._index[h] for h in hashes), dtype=np.int64, count=len(hashes)
        )
        unique_idx, inverse = np.unique(indices, return_inverse=True)
        dense_unique = self._f["dense"][unique_idx, :]
        dense = dense_unique[inverse]

        sparse = None
        if self.has_sparse:
            sparse_unique = self._f["sparse_json"].asstr()[unique_idx]
            sparse = [
                {int(k): v for k, v in json.loads(sparse_unique[i]).items()}
                for i in inverse
            ]

        return {"dense": dense, "sparse": sparse}


# =====================================================================
# Cache de embeddings token-level (schema ragged)
# =====================================================================


class TokenEmbeddingCache:
    """
    Cache de embeddings token-level por texto. Schema "ragged":
    todos os embeddings concatenados em um único dataset 2D, com um
    array de offsets indicando o início/fim de cada texto.

    Estrutura:
        /hashes        (N,)         vlen utf-8     sha256 hex do texto
        /offsets       (N+1,)       int64          offsets em /tokens_flat
        /tokens_flat   (sum_T, D)   float32        embeddings token-level
                                                   concatenados (sem padding)

    Para recuperar o texto i: tokens_flat[offsets[i]:offsets[i+1]].
    A semântica das posições (qual é CLS, qual é SEP, etc) é responsabilidade
    de quem usa — esta classe só armazena matrizes (T, D) arbitrárias.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: h5py.File | None = h5py.File(self.path, "a")
        if "hashes" in self._f:
            hashes = self._f["hashes"].asstr()[:]
            offsets = self._f["offsets"][:]
            self._index: dict[str, tuple[int, int]] = {
                h: (int(offsets[i]), int(offsets[i + 1]))
                for i, h in enumerate(hashes)
            }
            self._total_tokens = int(offsets[-1]) if len(offsets) > 0 else 0
        else:
            self._index = {}
            self._total_tokens = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __len__(self):
        return len(self._index)

    def __contains__(self, text: str) -> bool:
        return _hash(text) in self._index

    def get(self, text: str) -> np.ndarray | None:
        rng = self._index.get(_hash(text))
        if rng is None:
            return None
        start, end = rng
        return self._f["tokens_flat"][start:end, :]

    def _ensure_datasets(self, dim: int):
        str_dtype = h5py.string_dtype(encoding="utf-8")
        if "tokens_flat" not in self._f:
            self._f.create_dataset(
                "tokens_flat", shape=(0, dim), maxshape=(None, dim),
                dtype="float32", chunks=True, compression="lzf",
            )
        if "hashes" not in self._f:
            self._f.create_dataset(
                "hashes", shape=(0,), maxshape=(None,),
                dtype=str_dtype, chunks=True,
            )
        if "offsets" not in self._f:
            # cria offsets já com sentinela inicial [0]
            self._f.create_dataset(
                "offsets", shape=(1,), maxshape=(None,),
                dtype="int64", chunks=True,
            )
            self._f["offsets"][0] = 0

    def put_many(self, items: list[tuple[str, np.ndarray]]):
        """Append em lote: [(texto, emb (T, D)), ...]. Faz flush ao final."""
        if not items:
            return
        dim = int(items[0][1].shape[1])
        self._ensure_datasets(dim)

        # concatena tudo o que vai ser gravado
        new_embs = np.concatenate(
            [emb.astype("float32", copy=False) for _, emb in items], axis=0
        )
        new_hashes = [_hash(t) for t in (txt for txt, _ in items)]
        sizes = np.asarray([emb.shape[0] for _, emb in items], dtype=np.int64)
        new_offsets = self._total_tokens + np.cumsum(sizes)  # offsets[N+1:]

        n_old = self._f["hashes"].shape[0]
        n_new = len(items)
        tok_old = self._f["tokens_flat"].shape[0]
        tok_new_total = tok_old + int(new_embs.shape[0])

        self._f["tokens_flat"].resize(tok_new_total, axis=0)
        self._f["tokens_flat"][tok_old:] = new_embs

        self._f["hashes"].resize(n_old + n_new, axis=0)
        self._f["hashes"][n_old:] = np.asarray(new_hashes, dtype=object)

        # offsets cresce em n_new posições (já tem o 0 inicial)
        self._f["offsets"].resize(n_old + n_new + 1, axis=0)
        self._f["offsets"][n_old + 1:] = new_offsets

        # atualiza estado em memória
        prev = self._total_tokens
        for i, h in enumerate(new_hashes):
            end = int(new_offsets[i])
            self._index[h] = (prev, end)
            prev = end
        self._total_tokens = int(new_offsets[-1])
        self._f.flush()
