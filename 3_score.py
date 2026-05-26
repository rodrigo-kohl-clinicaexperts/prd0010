import json
import argparse
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm

from scoring.cache import EmbeddingCache, TokenEmbeddingCache

DATA = Path(__file__).parent / "data"
CACHE_DIR = DATA / "embedding_cache"


def load_generations():
    """
    Lê o generations.jsonl, descarta erros, e filtra por config_hash VIGENTE.

    Passo 1: para cada modelo, o config_hash vigente é o da última linha desse
             modelo no arquivo (append cronológico -> última = mais recente).
    Passo 2: dedup por (id, model) ficando com a última ocorrência, mas só
             aceitando linhas cujo config_hash == vigente do modelo. Gerações
             de config antiga são descartadas e reportadas.
    """
    path = DATA / "generations.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode 2_generate.py primeiro.")

    raw = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # linha truncada por crash — ignora

    # passo 1: config_hash vigente por modelo = o da última linha do modelo
    vigente = {}
    for r in raw:
        vigente[r["model"]] = r.get("config_hash")  # sobrescreve -> fica o último
    print("Config vigente por modelo (detectada do arquivo):")
    for model, h in vigente.items():
        print(f"  {model}: config_hash = {h}")

    # passo 2: dedup por (id, model), aceitando só o config_hash vigente
    dedup = {}
    descartados_config = 0
    for r in raw:
        model = r["model"]
        if r.get("config_hash") != vigente.get(model):
            descartados_config += 1
            continue  # geração de config antiga — descarta
        dedup[(r["id"], model)] = r  # última ocorrência válida vence

    if descartados_config:
        print(f"Filtro de config: {descartados_config} geração(ões) de config "
              f"antiga descartada(s).")

    rows = list(dedup.values())
    # descarta gerações que falharam
    ok = [r for r in rows if not str(r.get("generation", "")).startswith("__ERRO__")]
    erros = len(rows) - len(ok)
    if erros:
        print(f"Aviso: {erros} geração(ões) com erro foram ignoradas.")

    # avisa se algum modelo ficou sem nenhuma geração válida
    modelos_com_dados = {r["model"] for r in ok}
    modelos_no_arquivo = set(vigente.keys())
    sem_dados = modelos_no_arquivo - modelos_com_dados
    if sem_dados:
        print(f"AVISO: modelo(s) sem geração válida na config atual: "
              f"{sorted(sem_dados)}")
        print(f"  rode 2_generate.py para esse(s) modelo(s) antes de pontuar.")

    return ok


def load_scored():
    """Set de (id, model) já pontuados no scores.jsonl (para retomada)."""
    path = DATA / "scores.jsonl"
    scored = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scored[(r["id"], r["model"])] = r
    return scored


# --------------------------------------------------------------------------
# bge-m3 — produz dense, sparse e colbert vecs na mesma passada.
#
# O BGE-m3 computa internamente os três outputs no mesmo forward pass; pedir
# `return_colbert_vecs=True` em cima de dense+sparse não custa praticamente
# nada de inferência. Por isso o colbert F1 entra sem flag e sai junto.
# --------------------------------------------------------------------------

def _f1_greedy_cos(ref_emb: np.ndarray, cand_emb: np.ndarray) -> float:
    """F1 greedy de cosine similarity entre embeddings token-level."""
    if ref_emb is None or cand_emb is None:
        return 0.0
    if ref_emb.shape[0] == 0 or cand_emb.shape[0] == 0:
        return 0.0

    ref_norm = ref_emb / (np.linalg.norm(ref_emb, axis=-1, keepdims=True) + 1e-12)
    cand_norm = cand_emb / (np.linalg.norm(cand_emb, axis=-1, keepdims=True) + 1e-12)

    sim = cand_norm @ ref_norm.T  # (T_cand, T_ref)
    P = float(sim.max(axis=1).mean())
    R = float(sim.max(axis=0).mean())
    if P + R == 0:
        return 0.0
    return 2 * P * R / (P + R)


def score_bge(rows, append_every=256):
    from FlagEmbedding import BGEM3FlagModel

    refs = [r["reference"] for r in rows]
    gens = [r["generation"] for r in rows]

    model_holder = {"m": None}

    def _load_model():
        if model_holder["m"] is None:
            print("Carregando BAAI/bge-m3 ...")
            model_holder["m"] = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        return model_holder["m"]

    with TokenEmbeddingCache(CACHE_DIR / "bge-m3-colbert.h5") as tc:
        print(f"  cache bge-colbert: {len(tc)} embedding(s) já em disco.")

        def encode_fn(texts):
            m = _load_model()
            out = m.encode(texts, return_dense=True, return_sparse=True,
                           return_colbert_vecs=True,
                           batch_size=16, max_length=512)
            # mesmo forward pass alimenta o token cache também
            embs = [np.asarray(v, dtype="float32") for v in out["colbert_vecs"]]
            tc.put_many(list(zip(texts, embs)))
            return {"dense": out["dense_vecs"], "sparse": out["lexical_weights"]}

        with EmbeddingCache(CACHE_DIR / "bge-m3.h5", has_sparse=True) as cache:
            print(f"  cache bge-m3: {len(cache)} embedding(s) já em disco.")
            ref_cached = cache.get_or_encode(refs, encode_fn, label="bge refs",
                                             append_every=append_every)
            gen_cached = cache.get_or_encode(gens, encode_fn, label="bge gens",
                                             append_every=append_every)

        ref_d = np.asarray(ref_cached["dense"])
        gen_d = np.asarray(gen_cached["dense"])
        # vetores já vêm normalizados pelo FlagEmbedding -> produto interno = cosseno
        dense_sim = np.sum(ref_d * gen_d, axis=1)

        # matching lexical = soma dos produtos dos pesos sobre tokens em comum
        # (replicado do BGEM3FlagModel.compute_lexical_matching_score para evitar
        # ter de carregar o modelo só por causa dessa função)
        def lex_score(a: dict, b: dict) -> float:
            if len(a) > len(b):
                a, b = b, a
            return sum(w * b[t] for t, w in a.items() if t in b)

        sparse_sim = [
            lex_score(rw, gw)
            for rw, gw in zip(ref_cached["sparse"], gen_cached["sparse"])
        ]

        # backfill: textos com dense cache hit (encode_fn nem foi chamado) mas
        # sem colbert vecs ainda — caso o usuário tenha caches antigos.
        missing: list[str] = []
        seen: set[str] = set()
        for t in (*refs, *gens):
            if t in seen:
                continue
            seen.add(t)
            if t not in tc:
                missing.append(t)
        if missing:
            total = len(missing)
            print(f"  cache bge-colbert: {total} texto(s) sem colbert vecs "
                  f"(dense cache hit, token cache miss). Encodando...")
            for cs in range(0, total, append_every):
                ce = min(cs + append_every, total)
                chunk = missing[cs:ce]
                m = _load_model()
                out = m.encode(chunk, return_dense=False, return_sparse=False,
                               return_colbert_vecs=True,
                               batch_size=16, max_length=512)
                embs = [np.asarray(v, dtype="float32") for v in out["colbert_vecs"]]
                tc.put_many(list(zip(chunk, embs)))
                print(f"  colbert-cache flush {ce}/{total}")

        f1s = [_f1_greedy_cos(tc.get(r), tc.get(c)) for r, c in zip(refs, gens)]

    for i, r in enumerate(rows):
        r["bge_dense"] = float(dense_sim[i])
        r["bge_sparse"] = float(sparse_sim[i])
        r["bge_colbert_f1"] = float(f1s[i])

    model_holder["m"] = None
    return rows


# --------------------------------------------------------------------------
# multilingual-e5-large — encoder denso independente
# --------------------------------------------------------------------------

def score_e5(rows, append_every=256):
    from sentence_transformers import SentenceTransformer

    refs = [r["reference"] for r in rows]
    gens = [r["generation"] for r in rows]

    model_holder = {"m": None}

    def _load_model():
        if model_holder["m"] is None:
            print("Carregando intfloat/multilingual-e5-large ...")
            model_holder["m"] = SentenceTransformer(
                "intfloat/multilingual-e5-large"
            )
        return model_holder["m"]

    def encode_fn(texts):
        m = _load_model()
        # tarefa simétrica -> prefixo "query: " aplicado dentro do encode
        # (a chave do cache é o texto cru, não o texto com prefixo)
        prefixed = [f"query: {t}" for t in texts]
        emb = m.encode(prefixed, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=True)
        return {"dense": emb}

    with EmbeddingCache(CACHE_DIR / "e5-large.h5", has_sparse=False) as cache:
        print(f"  cache e5-large: {len(cache)} embedding(s) já em disco.")
        ref_cached = cache.get_or_encode(refs, encode_fn, label="e5 refs",
                                         append_every=append_every)
        gen_cached = cache.get_or_encode(gens, encode_fn, label="e5 gens",
                                         append_every=append_every)

    ref_emb = np.asarray(ref_cached["dense"])
    gen_emb = np.asarray(gen_cached["dense"])
    sim = np.sum(ref_emb * gen_emb, axis=1)
    for i, r in enumerate(rows):
        r["e5_dense"] = float(sim[i])

    model_holder["m"] = None
    return rows


# --------------------------------------------------------------------------

def main(args):
    gens = load_generations()
    print(f"{len(gens)} gerações disponíveis (após dedup e descarte de erros).")

    scored = {} if args.rescore else load_scored()
    if args.rescore:
        print("--rescore: recalculando tudo do zero.")
    elif scored:
        print(f"Retomando: {len(scored)} (id, model) já pontuados.")

    # Linhas nunca pontuadas: pipeline completo (bge + e5).
    pending_dense = [r for r in gens if (r["id"], r["model"]) not in scored]

    # Backfill: linhas já com bge+e5 mas sem bge_colbert_f1 (scores.jsonl
    # gerado quando colbert era opt-in). Só precisam do pass colbert do BGE.
    pending_colbert_only: list[dict] = []
    for r in gens:
        key = (r["id"], r["model"])
        existing = scored.get(key)
        if existing is None or "bge_colbert_f1" in existing:
            continue
        merged = dict(existing)
        # reference/generation podem não estar no scored — vêm do gens
        merged["reference"] = r["reference"]
        merged["generation"] = r["generation"]
        pending_colbert_only.append(merged)

    print(f"{len(pending_dense)} pendente(s) de bge+e5.")
    if pending_colbert_only:
        print(f"{len(pending_colbert_only)} já com bge+e5, "
              f"backfill de colbert.")
    print()

    if not pending_dense and not pending_colbert_only:
        print("Nada novo a pontuar — encoders nem serão carregados.")
        return

    new_rows: list[dict] = []
    if pending_dense:
        new_rows = score_bge(pending_dense, append_every=args.append_every)
        new_rows = score_e5(new_rows, append_every=args.append_every)

    if pending_colbert_only:
        # score_bge será cache hit em dense/sparse, encoda só colbert
        # e mutará in-place setando bge_colbert_f1.
        score_bge(pending_colbert_only, append_every=args.append_every)

    # junta: o que já estava pontuado + o que acabou de ser (re)pontuado
    final = dict(scored)  # (id, model) -> registro
    for r in new_rows:
        final[(r["id"], r["model"])] = r
    for r in pending_colbert_only:
        final[(r["id"], r["model"])] = r

    out_path = DATA / "scores.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSalvo: {out_path}  ({len(final)} linhas)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rescore", action="store_true",
                   help="ignora o scores.jsonl existente e recalcula tudo")
    p.add_argument("--append-every", type=int, default=256,
                   help="tamanho do chunk de append no cache de embeddings "
                        "(menor = mais flush, mais resiliente a crash; "
                        "maior = menos I/O). default: 256")
    main(p.parse_args())