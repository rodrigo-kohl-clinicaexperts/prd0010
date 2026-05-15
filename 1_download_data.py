import re
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from datasets import load_dataset
from dotenv import  load_dotenv
load_dotenv()
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Filtro de qualidade
# --------------------------------------------------------------------------

# Frases que sinalizam resposta evasiva — não medem conhecimento clínico,
# então penalizariam injustamente modelos que dão respostas substantivas.
EVASIVE_PATTERNS = [
    r"procure?\s+(um|seu)\s+(especialista|m[ée]dico|ortopedista)",
    r"agende\s+(sua|uma)\s+consulta",
    r"consulte\s+(um|seu)\s+m[ée]dico",
    r"marque\s+uma\s+consulta",
    r"deve\s+ser\s+avaliad[oa]\s+por",
    r"^\s*sim[.,!]?\s*$",
    r"^\s*n[ãa]o[.,!]?\s*$",
]
EVASIVE_RE = re.compile("|".join(EVASIVE_PATTERNS), re.IGNORECASE)


def is_evasive(answer: str) -> bool:
    """Resposta curta E que se resume a um encaminhamento vazio -> evasiva.
    As duas condições juntas: uma resposta longa que por acaso menciona
    'consulte um médico' no fim ainda tem conteúdo clínico e NÃO é cortada."""
    a = answer.strip()
    if len(a.split()) < 25 and EVASIVE_RE.search(a):
        return True
    return False



# --------------------------------------------------------------------------
# Amostragem estratificada por question_type, proporcional (sem piso)
# --------------------------------------------------------------------------

def proportional_quotas(stratum_sizes, target_n):
    """
    Dado {question_type: tamanho_disponível} e um N alvo, devolve
    {question_type: cota}.

    Proporcional puro, via "largest remainder":
      1. cota ideal = tamanho/total * N  (número fracionário)
      2. cada estrato recebe a parte inteira (floor)
      3. as vagas que sobraram vão para os estratos com maiores restos decimais
      4. clamp: cota nunca passa do tamanho disponível do estrato
      5. déficit gerado pelo clamp é redistribuído entre estratos com folga,
         para a soma fechar exatamente em target_n

    Sem piso por estrato: com ~7 categorias de question_type bem distribuídas,
    nenhuma corre risco de sumir numa amostra de tamanho razoável.
    """
    estratos = [s for s, sz in stratum_sizes.items() if sz > 0]
    total = sum(stratum_sizes[s] for s in estratos)
    if total == 0:
        return {}
    target_n = min(target_n, total)  # não dá para pegar mais do que existe

    # passo 1-2: cotas ideais e parte inteira
    ideal = {s: stratum_sizes[s] / total * target_n for s in estratos}
    quota = {s: int(math.floor(ideal[s])) for s in estratos}
    # clamp inicial
    for s in estratos:
        quota[s] = min(quota[s], stratum_sizes[s])

    # passo 3: distribui as vagas restantes pelos maiores restos decimais
    faltam = target_n - sum(quota.values())
    ordem = sorted(
        estratos,
        key=lambda s: ideal[s] - math.floor(ideal[s]),
        reverse=True,
    )
    i = 0
    while faltam > 0 and i < len(ordem) * 100:  # guarda contra loop infinito
        s = ordem[i % len(ordem)]
        if quota[s] < stratum_sizes[s]:
            quota[s] += 1
            faltam -= 1
        i += 1

    # passo 5: se o clamp comeu muita vaga, redistribui entre quem tem folga
    if faltam > 0:
        for s in sorted(estratos, key=lambda s: stratum_sizes[s], reverse=True):
            while faltam > 0 and quota[s] < stratum_sizes[s]:
                quota[s] += 1
                faltam -= 1
            if faltam == 0:
                break

    return quota


def stratified_sample(pool, target_n, seed):
    """
    pool: lista de questões já filtradas e deduplicadas.
    Estrato = question_type.
    Devolve (amostra, sizes, quotas).
    """
    rng = random.Random(seed)

    # agrupa por tipo de questão
    strata = defaultdict(list)
    for item in pool:
        key = item["question_type"] or "?"
        strata[key].append(item)

    sizes = {k: len(v) for k, v in strata.items()}
    quotas = proportional_quotas(sizes, target_n)

    # amostra dentro de cada estrato (random.sample simples — a única
    # estratificação pedida é por question_type)
    sample = []
    for key, items in strata.items():
        k = quotas.get(key, 0)
        if k > 0:
            if k >= len(items):
                sample.extend(items)
            else:
                sample.extend(rng.sample(items, k))

    # embaralha a ordem final (senão fica agrupado por estrato)
    rng.shuffle(sample)
    return sample, sizes, quotas


def incremental_sample(pool, existing_rows, target_n, seed):
    """
    Expande um eval_set existente até `target_n`, estratificado por
    question_type, PRESERVANDO todas as questões já existentes.

    pool:          todas as questões filtradas (universo disponível)
    existing_rows: lista de dicts do eval_set.parquet atual (a manter)
    target_n:      novo tamanho total desejado

    Para cada estrato:
      cota_nova   = cota proporcional para target_n
      ja_tem      = quantas questões daquele estrato já estão no eval_set
      deficit     = max(0, cota_nova - ja_tem)
      -> sorteia `deficit` questões NOVAS (ids ainda não usados) daquele estrato

    Se ja_tem > cota_nova (estrato já sobre-representado), mantém todas as
    existentes mesmo assim — nunca remove questão que já pode ter inferência —
    e avisa.

    Devolve (amostra_final, info) onde info traz os números por estrato para
    o relatório.
    """
    rng = random.Random(seed)

    ids_existentes = {r["id"] for r in existing_rows}

    # questões do pool que ainda NÃO estão no eval_set, agrupadas por estrato
    disponiveis = defaultdict(list)
    for item in pool:
        if item["id"] in ids_existentes:
            continue
        disponiveis[item["question_type"] or "?"].append(item)

    # quanto cada estrato já tem no eval_set existente
    ja_tem = Counter(r["question_type"] or "?" for r in existing_rows)

    # cotas proporcionais para o novo total — base é o pool inteiro
    pool_por_estrato = defaultdict(list)
    for item in pool:
        pool_por_estrato[item["question_type"] or "?"].append(item)
    sizes = {k: len(v) for k, v in pool_por_estrato.items()}
    cotas_novas = proportional_quotas(sizes, target_n)

    novas = []
    info = {}  # estrato -> (ja_tem, cota_nova, deficit, sorteadas)
    for estrato in sizes:
        tem = ja_tem.get(estrato, 0)
        cota = cotas_novas.get(estrato, 0)
        deficit = max(0, cota - tem)
        pool_disp = disponiveis.get(estrato, [])

        if deficit >= len(pool_disp):
            sorteadas = list(pool_disp)  # pega tudo que sobrou desse estrato
        else:
            sorteadas = rng.sample(pool_disp, deficit)

        novas.extend(sorteadas)
        info[estrato] = {
            "ja_tem": tem, "cota_nova": cota,
            "deficit": deficit, "sorteadas": len(sorteadas),
            "acima_da_cota": tem > cota,
        }

    # amostra final = existentes (intactas) + novas
    final = list(existing_rows) + novas
    rng.shuffle(final)
    return final, info


def decide_action(out_path, target_n, on_existing):
    """
    Detecta a situação do eval_set.parquet existente e resolve qual ação tomar:
    "generate" (do zero), "increment" (expande), ou "keep" (mantém e sai).

    on_existing: "ask" (padrão — detecta e pergunta), ou "increment" /
    "regenerate" / "keep" para rodar não-interativo sem pergunta.

    Devolve (acao, existing_rows) — existing_rows é None se não houver
    eval_set ou se a ação não precisar dele.
    """
    # nada no disco -> só dá para gerar do zero
    if not out_path.exists():
        if on_existing == "increment":
            raise FileNotFoundError(
                f"--on-existing increment pedido, mas {out_path} não existe. "
                f"Rode com regenerate (ou ask) para criar o eval_set inicial."
            )
        return "generate", None

    existing_df = pd.read_parquet(out_path)
    existing_rows = existing_df.to_dict(orient="records")
    n_atual = len(existing_rows)
    n_pedido = target_n if target_n is not None else float("inf")

    # flags de escape — pulam a pergunta
    if on_existing == "increment":
        if n_pedido <= n_atual:
            raise ValueError(
                f"--on-existing increment pedido, mas o eval_set atual "
                f"({n_atual}) não é menor que --max-rows ({target_n}). "
                f"Não há o que incrementar."
            )
        return "increment", existing_rows
    if on_existing == "regenerate":
        _aviso_regerar(n_atual)
        return "generate", None
    if on_existing == "keep":
        print(f"  --on-existing keep: mantendo o eval_set atual "
              f"({n_atual} questões). Nada a fazer.")
        return "keep", existing_rows

    # on_existing == "ask" -> detecta a situação e pergunta a coisa certa
    print(f"\n  Já existe um eval_set.parquet com {n_atual} questões "
          f"(você pediu {target_n}).")

    if n_pedido > n_atual:
        # menor que o pedido -> dá para incrementar
        print(f"  O eval_set atual é MENOR que o pedido.")
        print(f"    [i] incrementar  — adiciona {target_n - n_atual} questões, "
              f"preserva as {n_atual} atuais (e o generations.jsonl alinhado)")
        print(f"    [r] regerar      — amostra nova do zero (muda os ids)")
        print(f"    [k] manter       — não faz nada, sai")
        escolha = input("  escolha [i/r/k]: ").strip().lower()
        if escolha == "i":
            return "increment", existing_rows
        elif escolha == "r":
            _aviso_regerar(n_atual)
            return "generate", None
        else:
            print("  mantendo o eval_set atual.")
            return "keep", existing_rows
    else:
        # maior ou igual ao pedido -> incrementar não se aplica
        rel = "MAIOR" if n_pedido < n_atual else "do mesmo tamanho"
        print(f"  O eval_set atual é {rel} que o pedido — "
              f"não dá para incrementar preservando a estratificação.")
        print(f"    [r] regerar  — amostra nova do zero com {target_n} (muda os ids)")
        print(f"    [k] manter   — fica com o eval_set atual de {n_atual}")
        escolha = input("  escolha [r/k]: ").strip().lower()
        if escolha == "r":
            _aviso_regerar(n_atual)
            return "generate", None
        else:
            print(f"  mantendo o eval_set atual ({n_atual} questões).")
            return "keep", existing_rows


def _aviso_regerar(n_atual):
    """Aviso sobre o impacto de regerar no generations.jsonl."""
    print()
    print("  " + "!" * 60)
    print(f"  ATENÇÃO: regerar cria uma amostra nova, com IDS DIFERENTES.")
    print(f"  O generations.jsonl atual (se existir) ficará DESALINHADO — as")
    print(f"  inferências antigas são de ids que podem não estar no novo")
    print(f"  eval_set. Considere renomear data/generations.jsonl e")
    print(f"  data/scores.jsonl antes de seguir, para não misturar.")
    print("  " + "!" * 60)


# --------------------------------------------------------------------------

def main(args):
    print("Baixando AKCIT/MedPT (split train)...")
    ds = load_dataset("AKCIT/MedPT", split="train")
    print(f"  dataset bruto: {len(ds):,} linhas")

    # --- filtro de qualidade + dedup por pergunta ---
    seen_questions = set()
    pool = []
    dropped = {"vazio": 0, "muito_curta": 0, "muito_longa": 0, "evasiva": 0, "duplicada": 0}
    for row in ds:
        q = (row["question"] or "").strip()
        a = (row["answer"] or "").strip()
        if not q or not a:
            dropped["vazio"] += 1
            continue
        n = len(a.split())
        if n < args.min_answer_tokens:
            dropped["muito_curta"] += 1
            continue
        if n > args.max_answer_tokens:
            dropped["muito_longa"] += 1
            continue
        if is_evasive(a):
            dropped["evasiva"] += 1
            continue
        qkey = q.lower()
        if qkey in seen_questions:
            dropped["duplicada"] += 1
            continue
        seen_questions.add(qkey)
        pool.append({
            "id": row["id"],
            "question": q,
            "answer": a,
            "condition": (row["condition"] or "").strip(),
            "medical_specialty": (row["medical_specialty"] or "").strip(),
            "question_type": (row["question_type"] or "").strip(),
        })
    total_dropped = sum(dropped.values())
    print(f"  após filtro de qualidade + dedup: {len(pool):,} questões "
          f"({total_dropped:,} descartadas)")
    print(f"    vazio={dropped['vazio']:,}  "
          f"muito_curta={dropped['muito_curta']:,}  "
          f"muito_longa={dropped['muito_longa']:,}  "
          f"evasiva={dropped['evasiva']:,}  "
          f"duplicada={dropped['duplicada']:,}")

    n_especialidades = len({i["medical_specialty"] or "?" for i in pool})
    print(f"  especialidades distintas no pool: {n_especialidades}")

    # --- decide o que fazer: gerar do zero, incrementar, ou manter ---
    out_path = OUT / "eval_set.parquet"
    incr_info = None

    acao, existing_rows = decide_action(out_path, args.max_rows, args.on_existing)

    if acao == "keep":
        # mantém o eval_set atual intacto — não regrava nada, sai
        print("\nNada alterado. eval_set.parquet mantido como estava.")
        return

    if acao == "increment":
        sample, incr_info = incremental_sample(
            pool, existing_rows, args.max_rows, args.seed
        )
        print(f"\n  MODO INCREMENTAL (seed={args.seed})")
        print(f"  amostra final: {len(sample)} questões "
              f"({len(sample) - len(existing_rows)} novas, "
              f"{len(existing_rows)} preservadas)")

    elif acao == "generate":
        if args.max_rows and args.max_rows < len(pool):
            sample, sizes, quotas = stratified_sample(
                pool, args.max_rows, args.seed
            )
            print(f"\n  amostragem estratificada por question_type "
                  f"(seed={args.seed})")
            print(f"  proporcional puro (sem piso por estrato)")
            print(f"  amostra final: {len(sample):,} questões")
        else:
            sample = pool
            print(f"\n  usando o pool inteiro: {len(sample):,} questões "
                  f"(max_rows={args.max_rows} não restringe)")

    # --- relatório do modo incremental (déficit por estrato) ---
    if incr_info is not None:
        print("\n  === expansão incremental por estrato ===")
        print(f"  {'question_type':<35} {'tinha':>6} {'cota':>6} "
              f"{'+novas':>7}")
        for estrato, d in sorted(incr_info.items()):
            marca = "  <- acima da cota" if d["acima_da_cota"] else ""
            print(f"  {estrato:<35} {d['ja_tem']:>6} {d['cota_nova']:>6} "
                  f"{d['sorteadas']:>7}{marca}")
        acima = [e for e, d in incr_info.items() if d["acima_da_cota"]]
        if acima:
            print(f"  NOTA: estrato(s) acima da cota proporcional: {acima}")
            print(f"  (mantidas todas as existentes — nunca se remove questão "
                  f"que já pode ter inferência)")

    # --- relatórios de balanceamento (para você conferir) ---
    print("\n  === checagem de balanceamento da amostra ===")

    # distribuição por question_type — comparação pool vs amostra lado a lado
    # (é a coluna estratificada; aqui se vê se a proporção foi preservada)
    pool_qt = Counter(i["question_type"] for i in pool)
    samp_qt = Counter(i["question_type"] for i in sample)
    print(f"\n  por question_type (estrato — pool vs amostra):")
    for qt, _ in pool_qt.most_common():
        p_pool = 100 * pool_qt[qt] / len(pool)
        p_samp = 100 * samp_qt.get(qt, 0) / len(sample) if sample else 0
        n_samp = samp_qt.get(qt, 0)
        print(f"    {qt:<35} pool={p_pool:5.1f}%  "
              f"amostra={p_samp:5.1f}%  ({n_samp} questões)")

    # confirma que nenhum tipo de questão sumiu
    faltando = set(pool_qt) - set(samp_qt)
    if faltando:
        print(f"  AVISO: tipo(s) de questão ausente(s) na amostra: {faltando}")
    else:
        print(f"  OK: todos os {len(pool_qt)} tipos de questão presentes na amostra.")

    # as outras 2 colunas saem nos dados mas NÃO foram estratificadas —
    # mostramos a distribuição só a título informativo
    cond = Counter(i["condition"] for i in sample)
    print(f"\n  (informativo, não estratificado) por condition — top 10 "
          f"de {len(cond)}:")
    for val, n in cond.most_common(10):
        print(f"    {n:5d}  ({100*n/len(sample):4.1f}%)  {val}")

    esp = Counter(i["medical_specialty"] for i in sample)
    print(f"\n  (informativo, não estratificado) por medical_specialty — "
          f"top 10 de {len(esp)}:")
    for val, n in esp.most_common(10):
        print(f"    {n:5d}  ({100*n/len(sample):4.1f}%)  {val[:50]}")

    # --- salva ---
    # ordem de colunas explícita (sample é lista de dicts; garante consistência)
    cols = ["id", "question", "answer", "condition",
            "medical_specialty", "question_type"]
    df = pd.DataFrame(sample, columns=cols)

    df.to_parquet(out_path, index=False)
    print(f"\nSalvo: {out_path}  ({len(df):,} questões)")

    # preview em CSV só para inspeção a olho — NÃO é usado pelo pipeline
    preview_path = OUT / "eval_set_preview.csv"
    df.head(50).to_csv(preview_path, index=False)
    print(f"Preview (50 linhas, só para conferir): {preview_path}")

    print(f"Reprodutível: rode de novo com --seed {args.seed} para a mesma amostra.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-rows", type=int, default=1000,
                   help="tamanho da amostra estratificada (0 = usa o pool inteiro)")
    p.add_argument("--seed", type=int, default=42,
                   help="seed da amostragem — mesma seed reproduz a mesma amostra")
    p.add_argument("--on-existing", choices=["ask", "increment", "regenerate", "keep"],
                   default="ask",
                   help="o que fazer se já existir um eval_set.parquet: "
                        "'ask' (padrão) detecta a situação e pergunta; "
                        "'increment' expande sem perguntar; "
                        "'regenerate' gera do zero sem perguntar; "
                        "'keep' mantém o atual e sai. As 3 últimas são para "
                        "rodar não-interativo (pipeline encadeado, etc.)")
    p.add_argument("--min-answer-tokens", type=int, default=15,
                   help="descarta respostas de referência mais curtas que isso")
    p.add_argument("--max-answer-tokens", type=int, default=400,
                   help="descarta respostas de referência mais longas que isso")
    args = p.parse_args()
    if args.max_rows == 0:
        args.max_rows = None
    main(args)