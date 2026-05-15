import os
import json
import asyncio
import hashlib
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from makers import make_openai, make_openrouter, make_runpod, make_anthropic, make_gemini

load_dotenv()  # carrega o .env para o ambiente antes de ler as variáveis

DATA = Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Você é um assistente médico. Responda à pergunta do paciente de forma "
    "clara, correta e objetiva, em português do Brasil. Seja conciso e direto "
    "— responda o essencial sem se estender desnecessariamente. Responda "
    "apenas com a resposta, sem preâmbulos."
)

# kind define qual maker usar e qual semaphore/concorrência aplicar.
#   openai     — AsyncOpenAI nativo (api.openai.com, OPENAI_API_KEY)
#   openrouter — AsyncOpenAI apontado para OpenRouter (OpenAI-compatible)
#   runpod     — AsyncOpenAI apontado para endpoint vLLM do RunPod
#   anthropic  — SDK nativo da Anthropic (ANTHROPIC_API_KEY)
#   gemini     — SDK nativo do Google GenAI (GEMINI_API_KEY)
MODELS = {
    "gpt-5.2": {
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5.2",
        "kind": "openai",
    },
    "gpt-5.4-mini": {
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5.4-mini",
        "kind": "openai",
    },
    "gpt-5.4": {
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5.4",
        "kind": "openai",
    },
    "claude-sonnet-4": {
        # modelo direto via SDK Anthropic — sem passar pelo OpenRouter
        # verifique o ID exato em console.anthropic.com/docs/models
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-5",
        "kind": "anthropic",
    },
    "gemini-2.5-flash": {
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
        "kind": "gemini",
    },
    # MedGemma não está disponível na Gemini API pública — requer RunPod
    # "medgemma-4b-it": {
    #     "runpod_endpoint_env": "RUNPOD_MEDGEMMA_4B_ENDPOINT",
    #     "api_key_env": "RUNPOD_API_KEY",
    #     "model": "google/medgemma-4b-it",
    #     "kind": "runpod",
    # },
    # "medgemma-27b-text-it": {
    #     "runpod_endpoint_env": "RUNPOD_MEDGEMMA_27B_ENDPOINT",
    #     "api_key_env": "RUNPOD_API_KEY",
    #     "model": "google/medgemma-27b-text-it",
    #     "kind": "runpod",
    # },
}

MAX_TOKENS = 512
TEMPERATURE = 0.0

# Timeout por provider. RunPod tem cold start longo; demais usam 120 s.
REQUEST_TIMEOUT = {
    "openai":     120,
    "openrouter": 120,
    "runpod":     1200,
    "anthropic":  120,
    "gemini":     120,
}

MAX_RETRIES = 4
RETRY_BACKOFF = 5

# Concorrência padrão por provider.
DEFAULT_CONCURRENCY = {
    "openai":     make_openai.DEFAULT_CONCURRENCY,
    "openrouter": make_openrouter.DEFAULT_CONCURRENCY,
    "runpod":     make_runpod.DEFAULT_CONCURRENCY,
    "anthropic":  make_anthropic.DEFAULT_CONCURRENCY,
    "gemini":     make_gemini.DEFAULT_CONCURRENCY,
}

MAKERS = {
    "openai":     make_openai,
    "openrouter": make_openrouter,
    "runpod":     make_runpod,
    "anthropic":  make_anthropic,
    "gemini":     make_gemini,
}


# --------------------------------------------------------------------------
# Fingerprint da configuração (invalidação de cache)
# --------------------------------------------------------------------------

def config_fingerprint(model_id):
    """
    Hash curto que identifica a configuração que produz uma geração:
    system prompt + model_id + temperature + max_tokens. Se qualquer um muda,
    o hash muda e load_done deixa de reconhecer as gerações antigas, forçando
    a regeração só do que mudou.
    """
    payload = json.dumps(
        {
            "system_prompt": SYSTEM_PROMPT,
            "model_id": model_id,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Chamada ao modelo com retry/backoff
# --------------------------------------------------------------------------

async def call_model(client, name, spec, question):
    """
    Chama o modelo com retry/backoff. Delega a chamada real ao maker do kind.
    Devolve (texto, provedor).
    """
    kind = spec["kind"]
    maker = MAKERS[kind]
    last_err = None

    for tentativa in range(1, MAX_RETRIES + 1):
        try:
            return await maker.call_once(
                client, name, spec, question,
                SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS,
            )
        except Exception as e:
            # 400 e 404 são permanentes — não adianta retentar
            # openai/anthropic usam status_code; google-genai usa code
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status in (400, 404):
                raise
            last_err = e
            if status == 429:
                # usa retry-after do header se disponível; senão espera 60s
                retry_after = None
                resp = getattr(e, "response", None)
                if resp is not None:
                    retry_after = resp.headers.get("retry-after")
                espera = int(retry_after) + 1 if retry_after else 60
            else:
                espera = RETRY_BACKOFF * tentativa
            tqdm.write(
                f"  [{name}] tentativa {tentativa}/{MAX_RETRIES} falhou "
                f"({type(e).__name__}); aguardando {espera}s"
            )
            await asyncio.sleep(espera)

    raise RuntimeError(f"falhou após {MAX_RETRIES} tentativas: {last_err}")


# --------------------------------------------------------------------------
# Carga do eval_set e do que já foi gerado
# --------------------------------------------------------------------------

def load_eval_set():
    """Lê o eval_set.parquet produzido pela etapa 1."""
    path = DATA / "eval_set.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode 1_download_data.py primeiro."
        )
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def load_done(out_path, current_hashes):
    """
    Set de pares (id, model) já gerados COM A CONFIG ATUAL. Uma geração antiga
    só conta se o config_hash gravado bate com o hash atual daquele modelo.
    Ignora registros de erro (__ERRO__) e linhas truncadas por crash.
    """
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("generation", "")).startswith("__ERRO__"):
                continue
            model = r.get("model")
            if r.get("config_hash") != current_hashes.get(model):
                continue
            done.add((r["id"], model))
    return done


def purge_stale(out_path, current_hashes):
    """
    Reescreve o generations.jsonl mantendo só linhas cujo config_hash bate com
    a config atual (e que não são erro). Acionado por --purge-stale.
    """
    if not out_path.exists():
        print("  (nada a limpar — generations.jsonl não existe)")
        return
    kept, dropped = [], 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            model = r.get("model")
            is_err = str(r.get("generation", "")).startswith("__ERRO__")
            if (not is_err) and r.get("config_hash") == current_hashes.get(model):
                kept.append(line.rstrip("\n"))
            else:
                dropped += 1
    with open(out_path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
    print(f"  purge: {len(kept)} linhas mantidas, {dropped} removidas "
          f"(config antiga/erro).")


# --------------------------------------------------------------------------
# Geração de um modelo (async)
# --------------------------------------------------------------------------

async def run_one_question(client, name, spec, item, cfg_hash,
                           sem, fout, write_lock, pbar):
    async with sem:
        try:
            gen, provider = await call_model(client, name, spec, item["question"])
        except Exception as e:
            gen, provider = f"__ERRO__: {e}", "erro"
            tqdm.write(f"  erro definitivo em id={item['id']}: {e}")

    rec = {
        "id": int(item["id"]),
        "model": name,
        "question": item["question"],
        "reference": item["answer"],
        "generation": gen,
        "provider_used": provider,
        "config_hash": cfg_hash,
    }
    async with write_lock:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
    pbar.update(1)


async def run_one_model(name, spec, cfg_hash, pending, concurrency,
                        fout, write_lock, position=0):
    """
    Roda um modelo: as `pending` questões em paralelo, limitadas por semaphore.
    Cold start do RunPod: dispara 1 questão sozinha antes de liberar o resto.
    """
    kind = spec["kind"]
    maker = MAKERS[kind]
    timeout = REQUEST_TIMEOUT[kind]

    try:
        client = maker.make_client(spec, timeout)
    except Exception as e:
        tqdm.write(f"  ERRO ao iniciar backend de {name}: {e} — pulando.")
        return

    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(pending), desc=name, position=position, leave=True)

    try:
        fila = list(pending)

        if kind == "runpod" and fila:
            tqdm.write(f"  [{name}] aquecendo o endpoint (1 request inicial)...")
            primeiro = fila.pop(0)
            await run_one_question(client, name, spec, primeiro, cfg_hash,
                                   sem, fout, write_lock, pbar)

        tarefas = [
            run_one_question(client, name, spec, item, cfg_hash,
                             sem, fout, write_lock, pbar)
            for item in fila
        ]
        if tarefas:
            await asyncio.gather(*tarefas)
    finally:
        pbar.close()
        await maker.close_client(client)


async def run_track(kind, model_names, eval_set, current_hashes, done,
                    concurrency, fout, write_lock, position):
    """
    Uma "trilha": roda em SEQUÊNCIA os modelos do mesmo provider.
    As trilhas rodam em PARALELO entre si (ver main_async).
    """
    for name in model_names:
        spec = MODELS[name]
        cfg_hash = current_hashes[name]
        pending = [it for it in eval_set if (it["id"], name) not in done]

        tqdm.write(f"\n=== [{kind}] {name} ({spec['model']}) ===")
        if not pending:
            tqdm.write(f"  {name}: nada a fazer (tudo já gerado com a config atual).")
            continue
        tqdm.write(f"  {name}: {len(pending)} questões pendentes")

        await run_one_model(
            name, spec, cfg_hash, pending,
            concurrency[kind], fout, write_lock, position=position,
        )


# --------------------------------------------------------------------------

async def main_async(args):
    eval_set = load_eval_set()
    print(f"eval_set: {len(eval_set)} questões")

    models_to_run = args.models or list(MODELS.keys())
    print(f"Modelos a rodar: {models_to_run}")

    concurrency = {
        "openai":     args.concurrency_openai,
        "openrouter": args.concurrency_openrouter,
        "runpod":     args.concurrency_runpod,
        "anthropic":  args.concurrency_anthropic,
        "gemini":     args.concurrency_gemini,
    }
    print(
        f"Concorrência: OpenAI={concurrency['openai']}, "
        f"OpenRouter={concurrency['openrouter']}, "
        f"RunPod={concurrency['runpod']}, "
        f"Anthropic={concurrency['anthropic']}, "
        f"Gemini={concurrency['gemini']}"
    )

    current_hashes = {
        name: config_fingerprint(MODELS[name]["model"]) for name in MODELS
    }
    for name in models_to_run:
        print(f"  config_hash[{name}] = {current_hashes[name]}")

    out_path = DATA / "generations.jsonl"

    if args.purge_stale:
        print("\n--purge-stale: limpando linhas de config antiga...")
        purge_stale(out_path, current_hashes)

    done = load_done(out_path, current_hashes)
    if done:
        print(f"\nRetomando: {len(done)} gerações já feitas COM A CONFIG ATUAL "
              f"serão puladas.")

    tracks = {}
    for name in models_to_run:
        tracks.setdefault(MODELS[name]["kind"], []).append(name)
    print(f"\nTrilhas (rodam em paralelo entre si): "
          f"{ {k: v for k, v in tracks.items()} }")

    with open(out_path, "a", encoding="utf-8") as fout:
        write_lock = asyncio.Lock()
        track_coros = [
            run_track(kind, names, eval_set, current_hashes, done,
                      concurrency, fout, write_lock, position=i)
            for i, (kind, names) in enumerate(tracks.items())
        ]
        await asyncio.gather(*track_coros)

    print(f"\nConcluído. Resultados em {out_path}")
    final = load_done(out_path, current_hashes)
    esperado = len(eval_set) * len(models_to_run)
    print(f"Gerações bem-sucedidas (config atual): {len(final)}")
    if len(final) < esperado:
        print(f"  ({esperado - len(final)} faltando — rode de novo para "
              f"retomar só o que falta)")


def main(args):
    asyncio.run(main_async(args))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", choices=list(MODELS.keys()),
                   help="quais modelos rodar (padrão: todos)")
    p.add_argument("--purge-stale", action="store_true",
                   help="antes de rodar, remove do generations.jsonl as linhas "
                        "de config antiga (prompt/versão/parâmetros que mudaram)")
    p.add_argument("--concurrency-openai", type=int,
                   default=DEFAULT_CONCURRENCY["openai"],
                   help=f"chamadas em paralelo para OpenAI SDK "
                        f"(padrão: {DEFAULT_CONCURRENCY['openai']})")
    p.add_argument("--concurrency-openrouter", type=int,
                   default=DEFAULT_CONCURRENCY["openrouter"],
                   help=f"chamadas em paralelo para OpenRouter "
                        f"(padrão: {DEFAULT_CONCURRENCY['openrouter']})")
    p.add_argument("--concurrency-runpod", type=int,
                   default=DEFAULT_CONCURRENCY["runpod"],
                   help=f"chamadas em paralelo para RunPod — depende dos workers "
                        f"do endpoint (padrão: {DEFAULT_CONCURRENCY['runpod']})")
    p.add_argument("--concurrency-anthropic", type=int,
                   default=DEFAULT_CONCURRENCY["anthropic"],
                   help=f"chamadas em paralelo para Anthropic SDK "
                        f"(padrão: {DEFAULT_CONCURRENCY['anthropic']})")
    p.add_argument("--concurrency-gemini", type=int,
                   default=DEFAULT_CONCURRENCY["gemini"],
                   help=f"chamadas em paralelo para Gemini SDK "
                        f"(padrão: {DEFAULT_CONCURRENCY['gemini']})")
    main(p.parse_args())
