"""
LLM-as-judge: amostra estratificada das gerações de cada modelo e pede a
todos os outros modelos do pool (de kind DIFERENTE) para dar nota 0-10 sobre
coerência com a resposta de referência. Ensemble por padrão; cap opcional.

Output: data/judgments.jsonl (append-only, retomável).
Cache da amostra: data/judge_sample.json (estável e incrementável por seed).
"""
import os
import re
import json
import asyncio
import hashlib
import argparse
import importlib.util
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA = Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# Reusa MODELS / MAKERS / REQUEST_TIMEOUT / DEFAULT_CONCURRENCY do 2_generate
# (importação por importlib porque o nome do módulo começa com dígito).
# --------------------------------------------------------------------------

def _import_generate_module():
    spec = importlib.util.spec_from_file_location(
        "generate_module", Path(__file__).parent / "2_generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_GEN = _import_generate_module()
MODELS = _GEN.MODELS
MAKERS = _GEN.MAKERS
REQUEST_TIMEOUT = _GEN.REQUEST_TIMEOUT
DEFAULT_CONCURRENCY = _GEN.DEFAULT_CONCURRENCY
MAX_RETRIES = _GEN.MAX_RETRIES
RETRY_BACKOFF = _GEN.RETRY_BACKOFF

# --------------------------------------------------------------------------
# Configuração do juiz
# --------------------------------------------------------------------------

# Pool de juízes = MODELS (do 2_generate.py). Cada modelo julgado é avaliado
# por um ensemble dos outros de kind diferente. Única regra: mesmo kind não
# se julga. Cap padrão de 2 juízes por kind (evita que uma família grande
# domine o ensemble); ajustável via --max-judges-per-kind.

JUDGE_TEMPERATURE = 0.1
JUDGE_MAX_TOKENS = 320

JUDGE_SYSTEM_PROMPT = (
    "Você é um avaliador médico rigoroso e imparcial. Sua tarefa é julgar se "
    "a RESPOSTA DO MODELO é clinicamente coerente com a RESPOSTA DE "
    "REFERÊNCIA para a pergunta dada.\n\n"
    "Critérios:\n"
    "- Conteúdo: cobre os mesmos pontos médicos da referência?\n"
    "- Correção: sem erros factuais ou recomendações perigosas?\n"
    "- Adequação: responde de fato à pergunta?\n\n"
    "Rubrica (use a escala inteira, não só extremos):\n"
    "- 0: incorreta, perigosa ou totalmente irrelevante.\n"
    "- 3: parcial com erros importantes ou omissões graves.\n"
    "- 5: captura algum ponto, mas falta substância.\n"
    "- 7: alinhada na essência, divergências menores aceitáveis.\n"
    "- 10: clinicamente equivalente à referência (não precisa ser idêntica em "
    "palavras).\n\n"
    "Responda EXCLUSIVAMENTE com um JSON neste formato, sem texto antes nem "
    "depois, sem cercas de código:\n"
    '{"score": <inteiro 0-10>, "rationale": "<1-2 frases concisas em pt-BR>"}'
)

JUDGE_USER_TEMPLATE = (
    "PERGUNTA:\n{question}\n\n"
    "RESPOSTA DE REFERÊNCIA:\n{reference}\n\n"
    "RESPOSTA DO MODELO A JULGAR:\n{generation}\n\n"
    "Avalie e responda apenas com o JSON especificado."
)


def config_fingerprint() -> str:
    """Hash da config do juiz: muda → regera. Não inclui judge_model porque
    juízes diferentes geram entradas distintas (são chave, não invalidador).
    Também não inclui MODELS — adicionar/remover modelos não invalida
    julgamentos antigos, só muda o conjunto de pares a fazer."""
    payload = json.dumps({
        "system_prompt": JUDGE_SYSTEM_PROMPT,
        "user_template": JUDGE_USER_TEMPLATE,
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pick_judges(judged_model: str, max_per_kind: int | None = None) -> list[str]:
    """Modelos do MODELS com kind != julgado.
    Ordenação determinística: alfabética dentro de cada kind, kinds em ordem
    alfabética. `max_per_kind` capa quantos modelos por kind (ex: 2 evita
    que 3 gpts dominem o ensemble). None ou <=0 = sem cap."""
    judged_kind = MODELS[judged_model]["kind"]
    by_kind: dict[str, list[str]] = defaultdict(list)
    for name, spec in sorted(MODELS.items()):
        if spec["kind"] != judged_kind:
            by_kind[spec["kind"]].append(name)

    elegiveis: list[str] = []
    for kind in sorted(by_kind):
        membros = by_kind[kind]
        if max_per_kind is not None and max_per_kind > 0:
            membros = membros[:max_per_kind]
        elegiveis.extend(membros)

    if not elegiveis:
        raise RuntimeError(
            f"sem juiz elegível para '{judged_model}' (kind={judged_kind}) — "
            f"todos os modelos em MODELS são do mesmo kind")
    return elegiveis


# --------------------------------------------------------------------------
# Amostra estratificada determinística + incrementável
# --------------------------------------------------------------------------

def stratified_sample(eval_set, qtype_by_id, sample_size, seed):
    """
    Amostra estratificada por question_type, proporcional ao tamanho de cada
    estrato. Determinística por seed; aumentar sample_size é ADITIVO (os IDs
    antigos permanecem porque o shuffle por estrato é estável).
    """
    by_type = defaultdict(list)
    for it in eval_set:
        qt = qtype_by_id.get(it["id"], "?")
        by_type[qt].append(int(it["id"]))

    total = sum(len(v) for v in by_type.values())
    if sample_size > total:
        raise ValueError(
            f"sample_size={sample_size} > {total} questões disponíveis")

    rng = np.random.default_rng(seed)
    sampled_ids = []
    detalhe = {}
    for qt in sorted(by_type):  # ordem estável dos estratos
        ids_sorted = sorted(by_type[qt])
        # cada estrato tem seu próprio shuffle pra não vazar viés entre eles
        perm = rng.permutation(len(ids_sorted))
        shuffled = [ids_sorted[i] for i in perm]
        take = max(1, round(sample_size * len(ids_sorted) / total))
        chosen = shuffled[:take]
        sampled_ids.extend(chosen)
        detalhe[qt] = {"disponivel": len(ids_sorted), "amostrado": len(chosen)}

    return sampled_ids, detalhe


def load_or_build_sample(eval_set, qtype_by_id, sample_size, seed, force):
    """
    Cache da amostra em data/judge_sample.json. Se o cache existe com mesmo
    seed e sample_size compatível, reusa; senão regenera. --force ignora.
    """
    path = DATA / "judge_sample.json"
    if not force and path.exists():
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("seed") == seed and cached.get("sample_size") == sample_size:
            print(f"  reusando amostra cacheada: {len(cached['ids'])} IDs "
                  f"(seed={seed}, sample_size={sample_size})")
            return cached["ids"], cached["por_tipo"]
        if cached.get("seed") == seed and cached.get("sample_size", 0) < sample_size:
            print(f"  amostra cacheada tem {cached['sample_size']} < pedido "
                  f"({sample_size}); regerando com mesmo seed (os IDs antigos "
                  f"continuam no novo conjunto — shuffle é estável).")
        else:
            print(f"  cache existente com seed/tamanho diferentes — regerando.")

    ids, detalhe = stratified_sample(eval_set, qtype_by_id, sample_size, seed)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "sample_size": sample_size,
            "ids": ids,
            "por_tipo": detalhe,
        }, f, ensure_ascii=False, indent=2)
    print(f"  amostra salva em {path}: {len(ids)} IDs")
    return ids, detalhe


# --------------------------------------------------------------------------
# IO: eval_set + generations + judgments existentes
# --------------------------------------------------------------------------

def load_eval_set():
    path = DATA / "eval_set.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode 1_download_data.py.")
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def load_question_types():
    path = DATA / "eval_set.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["id"], df["question_type"]))


def load_generations():
    """(id, model) -> registro completo da geração. Pula __ERRO__."""
    path = DATA / "generations.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode 2_generate.py.")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("generation", "")).startswith("__ERRO__"):
                continue
            out[(int(r["id"]), r["model"])] = r
    return out


def load_done_judgments(cfg_hash):
    """Set de (id, judged, judge) já julgados COM A CONFIG ATUAL."""
    path = DATA / "judgments.jsonl"
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("config_hash") != cfg_hash:
                continue
            if r.get("score") is None:  # erro de parse — vamos retentar
                continue
            done.add((int(r["id"]), r["judged_model"], r["judge_model"]))
    return done


# --------------------------------------------------------------------------
# Parse robusto do output do juiz
# --------------------------------------------------------------------------

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(text: str):
    """
    Tenta extrair {"score": int 0-10, "rationale": str}. Devolve
    (score|None, rationale|""). Tolera cercas de código markdown e texto extra.
    """
    if not text:
        return None, ""
    # remove cerca de código se vier
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
        clean = re.sub(r"\n?```\s*$", "", clean)
    # tenta parse direto
    obj = None
    try:
        obj = json.loads(clean)
    except json.JSONDecodeError:
        # tenta extrair o primeiro {...} guloso (cobre texto antes/depois)
        m = _JSON_OBJ_RE.search(clean)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    if not isinstance(obj, dict):
        return None, ""
    score = obj.get("score")
    rationale = obj.get("rationale", "")
    if isinstance(score, bool):  # bool é int em Python, blindar
        return None, str(rationale)
    if not isinstance(score, (int, float)):
        return None, str(rationale)
    score = int(round(score))
    if not (0 <= score <= 10):
        return None, str(rationale)
    return score, str(rationale)


# --------------------------------------------------------------------------
# Chamada ao juiz (retry copia o padrão do 2_generate)
# --------------------------------------------------------------------------

async def call_judge(client, judge_name, judge_spec, user_msg):
    kind = judge_spec["kind"]
    maker = MAKERS[kind]
    last_err = None
    for tentativa in range(1, MAX_RETRIES + 1):
        try:
            return await maker.call_once(
                client, judge_name, judge_spec, user_msg,
                JUDGE_SYSTEM_PROMPT, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS,
            )
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status in (400, 404):
                raise
            last_err = e
            if status == 429:
                resp = getattr(e, "response", None)
                retry_after = (resp.headers.get("retry-after")
                               if resp is not None else None)
                espera = int(retry_after) + 1 if retry_after else 60
            else:
                espera = RETRY_BACKOFF * tentativa
            tqdm.write(f"  [{judge_name}] tentativa {tentativa}/{MAX_RETRIES} "
                       f"falhou ({type(e).__name__}); aguardando {espera}s")
            await asyncio.sleep(espera)
    raise RuntimeError(f"juiz falhou após {MAX_RETRIES} tentativas: {last_err}")


async def run_one_judgment(client, judge_name, judge_spec, pair, cfg_hash,
                           sem, fout, write_lock, pbar):
    async with sem:
        user_msg = JUDGE_USER_TEMPLATE.format(
            question=pair["question"],
            reference=pair["reference"],
            generation=pair["generation"],
        )
        try:
            raw, provider = await call_judge(client, judge_name, judge_spec, user_msg)
            score, rationale = parse_judge_output(raw)
        except Exception as e:
            raw, provider = f"__ERRO__: {e}", "erro"
            score, rationale = None, ""
            tqdm.write(f"  erro id={pair['id']} judged={pair['judged_model']} "
                       f"judge={judge_name}: {e}")

    rec = {
        "id": pair["id"],
        "judged_model": pair["judged_model"],
        "judge_model": judge_name,
        "score": score,
        "rationale": rationale,
        "raw_judge_output": raw,
        "provider_used": provider,
        "config_hash": cfg_hash,
    }
    async with write_lock:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
    pbar.update(1)


# --------------------------------------------------------------------------
# Batch API — estado persistente em data/judge_batches.jsonl
# --------------------------------------------------------------------------

JUDGE_BATCHES_PATH = DATA / "judge_batches.jsonl"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_judge_batches() -> dict:
    """batch_id -> entry (última versão); append-only."""
    if not JUDGE_BATCHES_PATH.exists():
        return {}
    entries = {}
    with open(JUDGE_BATCHES_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries[r["batch_id"]] = r
    return entries


def append_judge_batch(entry: dict) -> None:
    with open(JUDGE_BATCHES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


def items_in_pending_judge_batches(batches: dict, cfg_hash: str) -> set:
    """(id, judged_model, judge_model) já num batch não-fetchado da config atual."""
    in_flight = set()
    for entry in batches.values():
        if entry["status"] in ("fetched", "failed"):
            continue
        if entry["config_hash"] != cfg_hash:
            continue
        judge = entry["judge_model"]
        for p in entry["pairs"]:
            in_flight.add((int(p["id"]), p["judged_model"], judge))
    return in_flight


async def submit_judge_batches(pairs_by_judge, cfg_hash, sample_size, seed):
    """Um batch por juiz. custom_id = 'item_<idx>' (índice na lista de pairs
    do entry); pairs_meta guarda (judged_model, id) por idx."""
    for judge_name, pairs in pairs_by_judge.items():
        spec = MODELS[judge_name]
        kind = spec["kind"]
        maker = MAKERS[kind]

        if not maker.supports_batch():
            print(f"  [{kind}] {judge_name}: provider sem batch — pulando "
                  f"({len(pairs)} pares ficam pra rodar em modo sync).")
            continue
        if not pairs:
            print(f"  [{kind}] {judge_name}: nada a submeter.")
            continue

        items = []
        for idx, p in enumerate(pairs):
            user_msg = JUDGE_USER_TEMPLATE.format(
                question=p["question"],
                reference=p["reference"],
                generation=p["generation"],
            )
            items.append({"id": idx, "question": user_msg})

        client = maker.make_client(spec, REQUEST_TIMEOUT[kind])
        try:
            batch_id = await maker.submit_batch(
                client, judge_name, spec, items,
                JUDGE_SYSTEM_PROMPT, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS,
            )
        finally:
            await maker.close_client(client)

        # pairs_meta enxuto — question/reference/generation são recuperáveis
        # no fetch via generations.jsonl se precisar
        pairs_meta = [
            {"judged_model": p["judged_model"], "id": int(p["id"])}
            for p in pairs
        ]
        append_judge_batch({
            "batch_id": batch_id,
            "provider": kind,
            "judge_model": judge_name,
            "sample_size": sample_size,
            "seed": seed,
            "config_hash": cfg_hash,
            "pairs": pairs_meta,
            "submitted_at": _iso_now(),
            "status": "submitted",
            "fetched_at": None,
        })
        print(f"  [{kind}] {judge_name}: submetido {batch_id} "
              f"({len(items)} pares)")


async def poll_pending_judge_batches() -> dict:
    batches = load_judge_batches()
    pending = [bid for bid, e in batches.items()
               if e["status"] in ("submitted", "done")]
    if not pending:
        print("  nenhum batch de juiz pendente.")
        return batches

    for bid in pending:
        entry = batches[bid]
        kind = entry["provider"]
        judge_name = entry["judge_model"]
        spec = MODELS.get(judge_name)
        if spec is None:
            print(f"  {judge_name} [{bid}]: juiz não está mais em MODELS — pulando")
            continue
        maker = MAKERS[kind]
        client = maker.make_client(spec, REQUEST_TIMEOUT[kind])
        try:
            info = await maker.poll_batch(client, bid)
        finally:
            await maker.close_client(client)

        new_state = {
            "pending": "submitted",
            "done": "done",
            "failed": "failed",
        }[info["status"]]
        if new_state != entry["status"]:
            print(f"  {judge_name} [{bid}]: {entry['status']} → {new_state}")
            entry = dict(entry)
            entry["status"] = new_state
            append_judge_batch(entry)
            batches[bid] = entry
        else:
            print(f"  {judge_name} [{bid}]: {entry['status']}")
    return batches


async def fetch_completed_judge_batches():
    batches = load_judge_batches()
    todo = [(bid, e) for bid, e in batches.items() if e["status"] == "done"]
    if not todo:
        print("  nenhum batch pronto pra fetch.")
        return

    out_path = DATA / "judgments.jsonl"
    with open(out_path, "a", encoding="utf-8") as fout:
        for bid, entry in todo:
            kind = entry["provider"]
            judge_name = entry["judge_model"]
            spec = MODELS.get(judge_name)
            if spec is None:
                print(f"  {judge_name} [{bid}]: juiz não está mais em MODELS — pulando")
                continue
            maker = MAKERS[kind]
            cfg_hash = entry["config_hash"]
            pairs = entry["pairs"]

            client = maker.make_client(spec, REQUEST_TIMEOUT[kind])
            try:
                results = await maker.fetch_batch(client, bid)
            finally:
                await maker.close_client(client)

            n_ok, n_err = 0, 0
            for r in results:
                cid = r["custom_id"]
                try:
                    idx = int(cid.removeprefix("item_"))
                    pair = pairs[idx]
                except (ValueError, IndexError):
                    print(f"  custom_id inválido: {cid} — ignorando")
                    continue

                if r["error"]:
                    score, rationale = None, ""
                    raw = f"__ERRO__: {r['error']}"
                    n_err += 1
                else:
                    raw = r["generation"] or ""
                    score, rationale = parse_judge_output(raw)
                    if score is None:
                        n_err += 1
                    else:
                        n_ok += 1

                rec = {
                    "id": int(pair["id"]),
                    "judged_model": pair["judged_model"],
                    "judge_model": judge_name,
                    "score": score,
                    "rationale": rationale,
                    "raw_judge_output": raw,
                    "provider_used": r["provider_used"],
                    "config_hash": cfg_hash,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"  {judge_name} [{bid}]: fetched {n_ok} ok, {n_err} erro(s)")

            entry = dict(entry)
            entry["status"] = "fetched"
            entry["fetched_at"] = _iso_now()
            append_judge_batch(entry)


async def wait_until_all_done(poll_interval: int = 300):
    while True:
        batches = await poll_pending_judge_batches()
        ainda = [b for b in batches.values() if b["status"] == "submitted"]
        if not ainda:
            return
        print(f"  ainda processando: {len(ainda)} batch(es). "
              f"Próximo poll em {poll_interval}s.")
        await asyncio.sleep(poll_interval)


# --------------------------------------------------------------------------

async def run_one_judge(judge_name, judge_spec, pairs, cfg_hash, concurrency,
                        fout, write_lock, position=0):
    kind = judge_spec["kind"]
    maker = MAKERS[kind]
    timeout = REQUEST_TIMEOUT[kind]

    try:
        client = maker.make_client(judge_spec, timeout)
    except Exception as e:
        tqdm.write(f"  ERRO ao iniciar backend de {judge_name}: {e} — pulando.")
        return

    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(pairs), desc=f"judge:{judge_name}",
                position=position, leave=True)
    try:
        tarefas = [
            run_one_judgment(client, judge_name, judge_spec, p, cfg_hash,
                             sem, fout, write_lock, pbar)
            for p in pairs
        ]
        if tarefas:
            await asyncio.gather(*tarefas)
    finally:
        pbar.close()
        await maker.close_client(client)


# --------------------------------------------------------------------------

def build_pairs_by_judge(sample_ids, generations, models_to_judge,
                         judges_per_model, done, in_flight):
    """Monta {judge: [pair, ...]} pulando o que já está done ou in_flight."""
    pairs_by_judge = defaultdict(list)
    sem_geracao = 0
    for judged in models_to_judge:
        judges = judges_per_model[judged]
        for sid in sample_ids:
            gen_rec = generations.get((sid, judged))
            if gen_rec is None:
                sem_geracao += 1
                continue
            for judge in judges:
                key = (sid, judged, judge)
                if key in done or key in in_flight:
                    continue
                pairs_by_judge[judge].append({
                    "id": sid,
                    "judged_model": judged,
                    "question": gen_rec["question"],
                    "reference": gen_rec["reference"],
                    "generation": gen_rec["generation"],
                })
    return pairs_by_judge, sem_geracao


async def main_async(args):
    eval_set = load_eval_set()
    qtype_by_id = load_question_types()
    print(f"eval_set: {len(eval_set)} questões")

    cfg_hash = config_fingerprint()
    print(f"config_hash do juiz: {cfg_hash}")

    print(f"\n--- amostra (seed={args.seed}, sample_size={args.sample_size}) ---")
    sample_ids, detalhe = load_or_build_sample(
        eval_set, qtype_by_id, args.sample_size, args.seed, args.force_sample)
    for qt, d in sorted(detalhe.items(), key=lambda x: -x[1]["amostrado"]):
        print(f"  {str(qt)[:35]:<35} amostra={d['amostrado']:>4} / "
              f"{d['disponivel']:>4}")

    generations = load_generations()
    print(f"\ngenerations: {len(generations)} (id, model) válidos")

    models_to_judge = args.models or list(MODELS.keys())
    print(f"Modelos a julgar: {models_to_judge}")

    judges_per_model = {
        m: pick_judges(m, args.max_judges_per_kind)
        for m in models_to_judge
    }
    print("\nMapeamento julgado → juízes (ensemble; única regra: kind != julgado):")
    for m in models_to_judge:
        print(f"  {m:<22} ({MODELS[m]['kind']:<10}) → "
              f"{', '.join(judges_per_model[m])}")

    # --- modos batch ---
    if args.batch in ("submit", None):
        pass  # 'submit' tratado abaixo se args.batch == 'submit'

    if args.batch == "submit":
        print("\n=== batch submit ===")
        done = load_done_judgments(cfg_hash)
        in_flight = items_in_pending_judge_batches(load_judge_batches(), cfg_hash)
        if done:
            print(f"  já julgados (config atual): {len(done)}")
        if in_flight:
            print(f"  em batch pendente: {len(in_flight)}")
        pairs_by_judge, sem_geracao = build_pairs_by_judge(
            sample_ids, generations, models_to_judge, judges_per_model,
            done, in_flight)
        if sem_geracao:
            print(f"  aviso: {sem_geracao} (id, modelo) sem geração válida — pulados")
        await submit_judge_batches(pairs_by_judge, cfg_hash,
                                   args.sample_size, args.seed)
        return

    if args.batch == "poll":
        print("\n=== batch poll ===")
        await poll_pending_judge_batches()
        return

    if args.batch == "fetch":
        print("\n=== batch fetch ===")
        await poll_pending_judge_batches()
        await fetch_completed_judge_batches()
        return

    if args.batch == "wait":
        print("\n=== batch wait ===")
        await wait_until_all_done(poll_interval=args.poll_interval)
        await fetch_completed_judge_batches()
        return

    # --- modo sync (default) ---
    done = load_done_judgments(cfg_hash)
    if done:
        print(f"\nRetomando: {len(done)} julgamentos já feitos (config atual) "
              f"serão pulados.")
    pairs_by_judge, sem_geracao = build_pairs_by_judge(
        sample_ids, generations, models_to_judge, judges_per_model,
        done, in_flight=set())
    if sem_geracao:
        print(f"  aviso: {sem_geracao} (id, modelo) sem geração válida — pulados")

    total_pending = sum(len(p) for p in pairs_by_judge.values())
    print(f"\nTotal a julgar: {total_pending} pares "
          f"({len(pairs_by_judge)} juiz/juízes)")
    if total_pending == 0:
        print("Nada a fazer.")
        return

    concurrency = {
        "openai":     args.concurrency_openai,
        "openrouter": args.concurrency_openrouter,
        "runpod":     args.concurrency_runpod,
        "anthropic":  args.concurrency_anthropic,
        "gemini":     args.concurrency_gemini,
    }

    out_path = DATA / "judgments.jsonl"
    with open(out_path, "a", encoding="utf-8") as fout:
        write_lock = asyncio.Lock()
        trilhas = [
            run_one_judge(judge_name, MODELS[judge_name], pairs, cfg_hash,
                          concurrency[MODELS[judge_name]["kind"]],
                          fout, write_lock, position=i)
            for i, (judge_name, pairs) in enumerate(pairs_by_judge.items())
        ]
        await asyncio.gather(*trilhas)

    print(f"\nConcluído. Resultados em {out_path}")
    final = load_done_judgments(cfg_hash)
    esperado = sum(
        len(judges_per_model[judged])
        for judged in models_to_judge for sid in sample_ids
        if (sid, judged) in generations
    )
    print(f"Julgamentos bem-sucedidos (config atual): {len(final)} / "
          f"{esperado} esperados")
    if len(final) < esperado:
        print(f"  ({esperado - len(final)} faltando — rode de novo pra retomar)")


def main(args):
    asyncio.run(main_async(args))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-size", type=int, default=500,
                   help="quantas questões amostrar do eval_set "
                        "(estratificado por question_type). default: 500")
    p.add_argument("--seed", type=int, default=42,
                   help="seed da amostragem (incremento: aumentar sample-size "
                        "com mesmo seed mantém os IDs antigos). default: 42")
    p.add_argument("--force-sample", action="store_true",
                   help="ignora data/judge_sample.json e regenera a amostra")
    p.add_argument("--models", nargs="*", choices=list(MODELS.keys()),
                   help="quais modelos julgar (default: todos do pool)")
    p.add_argument("--max-judges-per-kind", type=int, default=2,
                   help="capa quantos juízes por kind (ex: 2 evita que 3 gpts "
                        "dominem o ensemble). default: 2. 0 ou negativo = sem cap.")
    p.add_argument("--batch", choices=["submit", "poll", "fetch", "wait"],
                   nargs="?", const="wait", default=None,
                   help="modo batch (50%% desconto, SLA 24-48h). "
                        "submit: envia pendentes; poll: checa status; "
                        "fetch: baixa prontos e dá append em judgments.jsonl; "
                        "wait: poll em loop até tudo pronto, depois fetch "
                        "(default quando --batch é passado sem valor).")
    p.add_argument("--poll-interval", type=int, default=300,
                   help="intervalo de poll (segundos) no modo --batch wait. "
                        "default: 300")
    p.add_argument("--concurrency-openai", type=int,
                   default=DEFAULT_CONCURRENCY["openai"])
    p.add_argument("--concurrency-openrouter", type=int,
                   default=DEFAULT_CONCURRENCY["openrouter"])
    p.add_argument("--concurrency-runpod", type=int,
                   default=DEFAULT_CONCURRENCY["runpod"])
    p.add_argument("--concurrency-anthropic", type=int,
                   default=DEFAULT_CONCURRENCY["anthropic"])
    p.add_argument("--concurrency-gemini", type=int,
                   default=DEFAULT_CONCURRENCY["gemini"])
    main(p.parse_args())
