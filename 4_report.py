import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA = Path(__file__).parent / "data"

# métricas onde "maior = melhor"
METRICS = ["bge_dense", "bge_sparse", "e5_dense", "bertscore_f1"]


def load_scores():
    path = DATA / "scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode 3_score.py primeiro.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_question_types():
    """
    Mapa id -> question_type, lido do eval_set.parquet. Usado para a quebra
    detalhada. (medical_specialty NÃO é usado: é texto multi-valor sujo.)
    """
    path = DATA / "eval_set.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["id"], df["question_type"]))


def mean_by_model(rows, metric):
    """Média de uma métrica por modelo. Ignora linhas sem a métrica."""
    acc = defaultdict(list)
    for r in rows:
        if metric in r:
            acc[r["model"]].append(r[metric])
    return {m: float(np.mean(v)) for m, v in acc.items() if v}


def ranking_from_means(means):
    """Dado {modelo: score}, devolve lista de modelos do melhor pro pior."""
    return [m for m, _ in sorted(means.items(), key=lambda x: -x[1])]


# --------------------------------------------------------------------------

def print_completeness(rows):
    """
    Checagem de completude: n de questões pontuadas por modelo. Se divergem
    muito, as médias não são diretamente comparáveis — avisa com destaque.
    Devolve o dict de counts.
    """
    print("\n" + "=" * 70)
    print("COMPLETUDE — n de questões pontuadas por modelo")
    print("=" * 70)

    counts = defaultdict(int)
    for r in rows:
        counts[r["model"]] += 1
    models = sorted(counts)
    for m in models:
        print(f"  {m:<22} n = {counts[m]}")

    if len(counts) > 1:
        vals = list(counts.values())
        n_min, n_max = min(vals), max(vals)
        # divergência relevante: o menor é < 90% do maior
        if n_min < 0.9 * n_max:
            print()
            print("  " + "!" * 60)
            print(f"  ATENÇÃO: os modelos têm n MUITO diferentes "
                  f"(de {n_min} a {n_max}).")
            print(f"  Médias de amostras de tamanhos diferentes NÃO são")
            print(f"  diretamente comparáveis. Provável causa: gerações com erro")
            print(f"  em algum modelo. Considere rodar 2_generate.py de novo")
            print(f"  (ele retoma só o que falta) antes de tirar conclusões.")
            print("  " + "!" * 60)
    return counts


def print_general_table(rows, present_metrics):
    print("\n" + "=" * 70)
    print("TABELA GERAL — média por modelo (maior = mais próximo da referência)")
    print("=" * 70)

    models = sorted({r["model"] for r in rows})
    header = f"{'modelo':<22}" + "".join(f"{m:>14}" for m in present_metrics)
    print(header)
    print("-" * len(header))

    means_by_metric = {m: mean_by_model(rows, m) for m in present_metrics}
    for model in models:
        line = f"{model:<22}"
        for m in present_metrics:
            v = means_by_metric[m].get(model)
            line += f"{v:>14.4f}" if v is not None else f"{'—':>14}"
        print(line)

    return means_by_metric


def print_rankings(means_by_metric):
    print("\n" + "=" * 70)
    print("RANKING por métrica")
    print("=" * 70)
    rankings = {}
    for metric, means in means_by_metric.items():
        rk = ranking_from_means(means)
        rankings[metric] = rk
        print(f"  {metric:<14}: " + " > ".join(rk))
    return rankings


def print_metric_agreement(rankings):
    """
    Teste de robustez: as métricas concordam no ranking?
    Correlação de Spearman entre os rankings. Perto de 1 = forte concordância.
    """
    print("\n" + "=" * 70)
    print("CONCORDÂNCIA ENTRE MÉTRICAS (Spearman entre rankings)")
    print("=" * 70)
    print("  ~1.0 = métricas concordam, resultado robusto")
    print("  baixo/negativo = ranking depende da métrica, conclusão frágil\n")

    metrics = list(rankings.keys())
    all_models = sorted({m for rk in rankings.values() for m in rk})
    pos = {}
    for metric, rk in rankings.items():
        pos[metric] = {model: rk.index(model) for model in rk}

    pares = []
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            m1, m2 = metrics[i], metrics[j]
            common = [mod for mod in all_models if mod in pos[m1] and mod in pos[m2]]
            if len(common) < 3:
                continue
            v1 = [pos[m1][mod] for mod in common]
            v2 = [pos[m2][mod] for mod in common]
            rho, _ = spearmanr(v1, v2)
            flag = "OK" if rho >= 0.7 else "ATENÇÃO: divergência"
            print(f"  {m1:<14} vs {m2:<14}: rho = {rho:+.3f}   [{flag}]")
            pares.append({"metric_a": m1, "metric_b": m2, "spearman": float(rho)})
    return pares


def print_by_question_type(rows, qtype_by_id, primary_metric):
    """Quebra por question_type — coluna limpa, 7 valores."""
    print("\n" + "=" * 70)
    print(f"QUEBRA POR QUESTION_TYPE (métrica: {primary_metric})")
    print("=" * 70)

    if not qtype_by_id:
        print("  (eval_set.parquet não encontrado — pulando quebra por tipo)")
        return {}

    # acc[tipo][modelo] = lista de scores
    acc = defaultdict(lambda: defaultdict(list))
    type_counts = defaultdict(int)
    for r in rows:
        if primary_metric not in r:
            continue
        qt = qtype_by_id.get(r["id"], "?")
        acc[qt][r["model"]].append(r[primary_metric])
        type_counts[qt] += 1

    models = sorted({r["model"] for r in rows})
    # ordena os tipos por frequência (maior primeiro)
    types_sorted = sorted(type_counts, key=lambda t: -type_counts[t])

    header = f"{'question_type':<32}" + "".join(f"{m:>16}" for m in models)
    print(header)
    print("-" * len(header))
    breakdown = {}
    for qt in types_sorted:
        line = f"{str(qt)[:31]:<32}"
        breakdown[qt] = {}
        for model in models:
            vals = acc[qt][model]
            if vals:
                mean_v = float(np.mean(vals))
                line += f"{mean_v:>16.4f}"
                breakdown[qt][model] = mean_v
            else:
                line += f"{'—':>16}"
        print(line)
    return breakdown


def collect_audit_info(rows):
    """config_hash e provider_used por modelo — para auditabilidade no report."""
    hashes = defaultdict(set)
    providers = defaultdict(set)
    for r in rows:
        if r.get("config_hash"):
            hashes[r["model"]].add(r["config_hash"])
        if r.get("provider_used"):
            providers[r["model"]].add(r["provider_used"])
    # converte sets para listas (json não serializa set)
    return (
        {m: sorted(h) for m, h in hashes.items()},
        {m: sorted(p) for m, p in providers.items()},
    )


def main(args):
    rows = load_scores()
    present_metrics = [m for m in METRICS if any(m in r for r in rows)]
    print(f"{len(rows)} linhas pontuadas. Métricas presentes: {present_metrics}")

    counts = print_completeness(rows)
    means_by_metric = print_general_table(rows, present_metrics)
    rankings = print_rankings(means_by_metric)
    agreement = print_metric_agreement(rankings)

    qtype_by_id = load_question_types()
    primary = "bge_dense" if "bge_dense" in present_metrics else (
        present_metrics[0] if present_metrics else None)
    breakdown = {}
    if primary:
        breakdown = print_by_question_type(rows, qtype_by_id, primary)

    config_hashes, providers = collect_audit_info(rows)

    # dump estruturado
    report = {
        "n_rows": len(rows),
        "n_by_model": dict(counts),
        "metrics": present_metrics,
        "means_by_metric": means_by_metric,
        "rankings": rankings,
        "metric_agreement_spearman": agreement,
        "breakdown_by_question_type": breakdown,
        "audit": {
            "config_hash_by_model": config_hashes,
            "provider_used_by_model": providers,
        },
    }
    out_path = DATA / "report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório estruturado salvo em {out_path}")

    print("\n" + "=" * 70)
    print("LEMBRETE METODOLÓGICO")
    print("=" * 70)
    print("""  Estas métricas medem PROXIMIDADE À RESPOSTA DE REFERÊNCIA, não
  qualidade clínica absoluta. O MedPT é Q&A de fórum: a referência é
  uma resposta humana entre várias possíveis. Para conclusões fortes,
  valide o ranking com o LLM-as-judge (6_llm_judge.py) num subconjunto.""")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    main(p.parse_args())