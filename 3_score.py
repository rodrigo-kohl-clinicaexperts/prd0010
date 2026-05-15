import json
import argparse
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm

DATA = Path(__file__).parent / "data"


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
# bge-m3 — dá dense E sparse na mesma passada
# --------------------------------------------------------------------------

def score_bge(rows):
    from FlagEmbedding import BGEM3FlagModel

    print("Carregando BAAI/bge-m3 ...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    refs = [r["reference"] for r in rows]
    gens = [r["generation"] for r in rows]

    print("  codificando referências e gerações (dense + sparse)...")
    # bge-m3 não precisa de prefixo
    ref_out = model.encode(refs, return_dense=True, return_sparse=True,
                           batch_size=16, max_length=512)
    gen_out = model.encode(gens, return_dense=True, return_sparse=True,
                           batch_size=16, max_length=512)

    # dense: cosseno (vetores já vêm normalizados pelo FlagEmbedding)
    ref_d = np.asarray(ref_out["dense_vecs"])
    gen_d = np.asarray(gen_out["dense_vecs"])
    dense_sim = np.sum(ref_d * gen_d, axis=1)

    # sparse: o próprio modelo calcula o score de matching lexical
    sparse_sim = []
    for rw, gw in zip(ref_out["lexical_weights"], gen_out["lexical_weights"]):
        sparse_sim.append(model.compute_lexical_matching_score(rw, gw))

    for i, r in enumerate(rows):
        r["bge_dense"] = float(dense_sim[i])
        r["bge_sparse"] = float(sparse_sim[i])

    del model
    return rows


# --------------------------------------------------------------------------
# multilingual-e5-large — encoder denso independente
# --------------------------------------------------------------------------

def score_e5(rows):
    from sentence_transformers import SentenceTransformer

    print("Carregando intfloat/multilingual-e5-large ...")
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    # tarefa simétrica -> prefixo "query: " nos dois lados
    refs = [f"query: {r['reference']}" for r in rows]
    gens = [f"query: {r['generation']}" for r in rows]

    print("  codificando referências e gerações...")
    ref_emb = model.encode(refs, batch_size=32, normalize_embeddings=True,
                           show_progress_bar=True)
    gen_emb = model.encode(gens, batch_size=32, normalize_embeddings=True,
                           show_progress_bar=True)

    sim = np.sum(ref_emb * gen_emb, axis=1)
    for i, r in enumerate(rows):
        r["e5_dense"] = float(sim[i])

    del model
    return rows


# --------------------------------------------------------------------------
# BERTScore — sanity check lexical (opcional)
# --------------------------------------------------------------------------

def score_bertscore(rows):
    from bert_score import score as bert_score

    print("Calculando BERTScore (modelo base multilíngue)...")
    cands = [r["generation"] for r in rows]
    refs = [r["reference"] for r in rows]
    # modelo base multilíngue; lang='pt' ajusta baseline
    P, R, F1 = bert_score(cands, refs, lang="pt", verbose=True,
                          model_type="bert-base-multilingual-cased")
    for i, r in enumerate(rows):
        r["bertscore_f1"] = float(F1[i])
    return rows


# --------------------------------------------------------------------------

def main(args):
    gens = load_generations()
    print(f"{len(gens)} gerações disponíveis (após dedup e descarte de erros).")

    scored = {} if args.rescore else load_scored()
    if args.rescore:
        print("--rescore: recalculando tudo do zero.")
    elif scored:
        print(f"Retomando: {len(scored)} (id, model) já pontuados serão pulados.")

    # o que falta pontuar
    pending = [r for r in gens if (r["id"], r["model"]) not in scored]
    print(f"{len(pending)} gerações a pontuar.\n")

    if not pending:
        print("Nada novo a pontuar — encoders nem serão carregados.")
        # ainda assim reescreve o arquivo (caso --rescore ou dedup tenha mudado algo)
        if args.rescore:
            pass  # nada a fazer mesmo
        return

    # só agora carrega os encoders (pesados) — e só se há trabalho
    rows = score_bge(pending)
    rows = score_e5(rows)
    if args.bertscore:
        rows = score_bertscore(rows)

    # junta: o que já estava pontuado + o que acabou de ser pontuado
    final = dict(scored)  # (id, model) -> registro
    for r in rows:
        final[(r["id"], r["model"])] = r

    out_path = DATA / "scores.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSalvo: {out_path}  ({len(final)} linhas)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bertscore", action="store_true",
                   help="também calcula BERTScore (mais lento)")
    p.add_argument("--rescore", action="store_true",
                   help="ignora o scores.jsonl existente e recalcula tudo")
    main(p.parse_args())