"""
Analisa data/judgments.jsonl (do 6_llm_judge.py):
- agrega scores em data/judge_report.json (média ensemble por modelo, quebra
  por question_type, concordância inter-juízes, dureza/calibração por juiz,
  correlação com ranking de embedding);
- reescreve data/report.html unindo painéis de embedding (de 5_visualize)
  com painéis do juiz no mesmo dashboard.

Roda depois de 4_report.py + 6_llm_judge.py.
"""
import json
import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr

DATA = Path(__file__).parent / "data"


def _import(module_path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_VIZ = _import(Path(__file__).parent / "5_visualize.py", "viz")


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def load_judgments(cfg_hash: str | None = None):
    """Carrega TUDO de judgments.jsonl. Filtra config_hash se fornecido."""
    path = DATA / "judgments.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode 6_llm_judge.py.")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cfg_hash and r.get("config_hash") != cfg_hash:
                continue
            rows.append(r)
    return rows


def load_question_types():
    path = DATA / "eval_set.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["id"], df["question_type"]))


def load_embedding_report():
    path = DATA / "report.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Agregações
# --------------------------------------------------------------------------

def per_question_ensemble(rows_scored):
    """{(id, judged_model): mean_score across all judges}. Já é o 'ensemble'."""
    bucket = defaultdict(list)
    for r in rows_scored:
        bucket[(int(r["id"]), r["judged_model"])].append(float(r["score"]))
    return {k: float(np.mean(v)) for k, v in bucket.items()}


def mean_by_judged_model(per_q):
    """Média do ensemble sobre todas as questões, por modelo julgado."""
    by_m = defaultdict(list)
    for (_id, m), s in per_q.items():
        by_m[m].append(s)
    return {m: float(np.mean(v)) for m, v in by_m.items()}


def breakdown_by_question_type(per_q, qtype_by_id):
    """{question_type: {judged_model: mean_score}}, baseado no ensemble por questão."""
    by = defaultdict(lambda: defaultdict(list))
    for (id_, m), s in per_q.items():
        qt = qtype_by_id.get(id_, "?")
        by[qt][m].append(s)
    return {qt: {m: float(np.mean(v)) for m, v in d.items()}
            for qt, d in by.items()}


def per_judge_stats(all_rows):
    """Auditoria de calibração de cada juiz: média, n, taxa de parse_fail."""
    by_judge = defaultdict(lambda: {"scores": [], "total": 0, "fails": 0})
    for r in all_rows:
        j = r["judge_model"]
        by_judge[j]["total"] += 1
        s = r.get("score")
        if s is None:
            by_judge[j]["fails"] += 1
        else:
            by_judge[j]["scores"].append(float(s))
    out = {}
    for j, d in by_judge.items():
        n = len(d["scores"])
        out[j] = {
            "n": n,
            "mean_score": float(np.mean(d["scores"])) if n else None,
            "std_score": float(np.std(d["scores"])) if n else None,
            "parse_fail_rate": d["fails"] / d["total"] if d["total"] else 0.0,
        }
    return out


def inter_judge_agreement(rows_scored):
    """
    Para cada par de juízes: Spearman entre os rankings de modelos julgados.
    Cada juiz produz {judged_model: mean_score}; correlacionamos os rankings.
    """
    by_judge = defaultdict(lambda: defaultdict(list))
    for r in rows_scored:
        by_judge[r["judge_model"]][r["judged_model"]].append(float(r["score"]))
    means_by_judge = {
        j: {m: float(np.mean(v)) for m, v in d.items()}
        for j, d in by_judge.items()
    }
    judges = sorted(means_by_judge)
    pares = []
    for i in range(len(judges)):
        for k in range(i + 1, len(judges)):
            j1, j2 = judges[i], judges[k]
            common = sorted(set(means_by_judge[j1]) & set(means_by_judge[j2]))
            if len(common) < 3:
                continue
            v1 = [means_by_judge[j1][m] for m in common]
            v2 = [means_by_judge[j2][m] for m in common]
            rho, _ = spearmanr(v1, v2)
            pares.append({
                "judge_a": j1, "judge_b": j2,
                "spearman": float(rho), "n_models": len(common),
            })
    return pares, means_by_judge


def cross_correlation_with_embedding(judge_means_by_model, embed_report):
    """
    Spearman entre o ranking do juiz (média ensemble) e o ranking de cada
    métrica de embedding. ~1 = juiz e embedding concordam → o ranking de
    embedding está validado por uma fonte de verdade diferente.
    """
    if not embed_report:
        return []
    out = []
    for metric, embed_means in embed_report.get("means_by_metric", {}).items():
        common = sorted(set(judge_means_by_model) & set(embed_means))
        if len(common) < 3:
            continue
        v_judge = [judge_means_by_model[m] for m in common]
        v_embed = [embed_means[m] for m in common]
        rho, _ = spearmanr(v_judge, v_embed)
        out.append({
            "metric": metric, "spearman": float(rho), "n_models": len(common),
        })
    return out


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------

def fig_mean_by_model(judge_means):
    if not judge_means:
        return None
    models = sorted(judge_means, key=lambda m: -judge_means[m])
    vals = [judge_means[m] for m in models]
    fig = go.Figure(data=go.Bar(
        x=models, y=vals,
        marker_color="steelblue",
        text=[f"{v:.2f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title="Juiz — média de score por modelo (ensemble, escala 0-10)",
        xaxis_title="modelo julgado",
        yaxis_title="score médio (0-10)",
        yaxis_range=[0, 10],
    )
    return fig


def fig_breakdown_heatmap(breakdown, judge_means):
    if not breakdown:
        return None
    qtypes = list(breakdown.keys())
    models = sorted({m for d in breakdown.values() for m in d})
    columns = qtypes + ["TOTAL"]
    z = [
        [breakdown[qt].get(m) for qt in qtypes] + [judge_means.get(m)]
        for m in models
    ]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=columns, y=models,
        colorscale="Viridis", zmin=0, zmax=10,
        text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title="score"),
    ))
    fig.add_vline(x=len(qtypes) - 0.5, line_width=2, line_color="white")
    fig.update_layout(
        title="Juiz — desempenho por question_type (TOTAL = média global)",
        xaxis_title="question_type",
        yaxis_title="modelo julgado",
    )
    return fig


def fig_inter_judge_heatmap(pairs):
    if not pairs:
        return None
    judges = sorted({p["judge_a"] for p in pairs} | {p["judge_b"] for p in pairs})
    idx = {j: i for i, j in enumerate(judges)}
    n = len(judges)
    matrix = [[None] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
    for p in pairs:
        i, k = idx[p["judge_a"]], idx[p["judge_b"]]
        matrix[i][k] = p["spearman"]
        matrix[k][i] = p["spearman"]
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=judges, y=judges,
        zmin=-1, zmax=1, colorscale="RdYlGn",
        text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in matrix],
        texttemplate="%{text}",
        colorbar=dict(title="Spearman"),
    ))
    fig.update_layout(
        title="Juiz — concordância entre juízes (Spearman entre rankings) — "
              "verde = ensemble coerente, vermelho = juízes discordam",
    )
    return fig


def fig_judge_calibration(stats):
    if not stats:
        return None
    judges = sorted(stats)
    means = [stats[j]["mean_score"] for j in judges]
    fails = [stats[j]["parse_fail_rate"] * 100 for j in judges]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="média de score atribuída", x=judges, y=means,
        marker_color="steelblue",
        text=[f"{v:.2f}" if v is not None else "" for v in means],
        textposition="outside",
        yaxis="y",
    ))
    fig.add_trace(go.Bar(
        name="parse_fail (%)", x=judges, y=fails,
        marker_color="crimson",
        text=[f"{v:.1f}%" for v in fails], textposition="outside",
        yaxis="y2",
    ))
    fig.update_layout(
        title="Juiz — calibração: dureza média (azul) e taxa de parse_fail (vermelho) por juiz",
        barmode="group",
        xaxis_title="juiz",
        yaxis=dict(title="score médio atribuído (0-10)", range=[0, 10]),
        yaxis2=dict(title="parse_fail (%)", overlaying="y", side="right",
                    range=[0, max(10, max(fails) * 1.2 if fails else 10)]),
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def fig_judge_vs_embedding(cross_corr):
    if not cross_corr:
        return None
    metrics = [c["metric"] for c in cross_corr]
    rhos = [c["spearman"] for c in cross_corr]
    cores = ["seagreen" if r >= 0.7 else "orange" if r >= 0.4 else "crimson"
             for r in rhos]
    fig = go.Figure(data=go.Bar(
        x=metrics, y=rhos, marker_color=cores,
        text=[f"{r:+.2f}" for r in rhos], textposition="outside",
    ))
    fig.update_layout(
        title="Juiz vs embedding — Spearman entre rankings (verde ≥0.7 = "
              "embedding validado pelo juiz)",
        xaxis_title="métrica de embedding",
        yaxis_title="Spearman vs ranking do juiz",
        yaxis_range=[-1, 1],
    )
    return fig


# --------------------------------------------------------------------------

def main(args):
    print("Carregando judgments.jsonl...")
    all_rows = load_judgments(cfg_hash=args.config_hash)
    if not all_rows:
        raise RuntimeError("nenhum julgamento encontrado (config_hash filtra?)")
    rows_scored = [r for r in all_rows if r.get("score") is not None]
    print(f"  total: {len(all_rows)} linhas, {len(rows_scored)} com score válido")

    qtype_by_id = load_question_types()
    embed_report = load_embedding_report()

    print("\nAgregando...")
    per_q = per_question_ensemble(rows_scored)
    judge_means = mean_by_judged_model(per_q)
    breakdown = breakdown_by_question_type(per_q, qtype_by_id)
    judge_stats = per_judge_stats(all_rows)
    inter_pairs, judge_x_model_means = inter_judge_agreement(rows_scored)
    cross_corr = cross_correlation_with_embedding(judge_means, embed_report)

    print("\n=== Média ensemble por modelo ===")
    for m, v in sorted(judge_means.items(), key=lambda x: -x[1]):
        print(f"  {m:<22} {v:>6.2f}")

    print("\n=== Calibração por juiz ===")
    for j, s in sorted(judge_stats.items()):
        mean = s["mean_score"]
        ms = f"{mean:.2f}" if mean is not None else "—"
        print(f"  {j:<22} n={s['n']:>5}  média={ms:<6}  "
              f"parse_fail={s['parse_fail_rate']*100:.1f}%")

    if inter_pairs:
        print("\n=== Concordância entre juízes (Spearman) ===")
        for p in inter_pairs:
            flag = "OK" if p["spearman"] >= 0.7 else "atenção"
            print(f"  {p['judge_a']:<22} vs {p['judge_b']:<22} "
                  f"rho={p['spearman']:+.3f} [{flag}]")

    if cross_corr:
        print("\n=== Juiz vs embedding (Spearman) ===")
        for c in cross_corr:
            flag = "OK" if c["spearman"] >= 0.7 else "atenção"
            print(f"  vs {c['metric']:<16} rho={c['spearman']:+.3f}  "
                  f"(n_modelos={c['n_models']}) [{flag}]")

    # ----- salva judge_report.json
    judge_report = {
        "n_judgments_total": len(all_rows),
        "n_judgments_scored": len(rows_scored),
        "n_questions": len({k[0] for k in per_q}),
        "n_models": len(judge_means),
        "mean_by_judged_model": judge_means,
        "breakdown_by_question_type": breakdown,
        "per_judge_stats": judge_stats,
        "inter_judge_agreement_spearman": inter_pairs,
        "judge_means_per_model_per_judge": judge_x_model_means,
        "cross_correlation_with_embedding_spearman": cross_corr,
    }
    out_json = DATA / "judge_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(judge_report, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório do juiz salvo em {out_json}")

    if args.no_html:
        return

    if embed_report is None:
        print("\nAVISO: data/report.json não existe — pulando rebuild do HTML "
              "(rode 4_report.py para incluir os painéis de embedding).")
        return

    # ----- monta figuras do juiz
    judge_figs = []
    for fig in (
        fig_mean_by_model(judge_means),
        fig_breakdown_heatmap(breakdown, judge_means),
        fig_inter_judge_heatmap(inter_pairs),
        fig_judge_calibration(judge_stats),
        fig_judge_vs_embedding(cross_corr),
    ):
        if fig is not None:
            judge_figs.append(fig)

    extra_header = (
        "<h2>LLM-as-judge — coerência clínica com a referência (score 0-10)</h2>"
        f"<div class='meta'>{len(rows_scored)} julgamentos sobre "
        f"{judge_report['n_questions']} questões, {len(judge_stats)} juízes "
        "(ensemble; cada questão = média entre juízes que a avaliaram).</div>"
    )
    extra_meta = f"juiz: {len(rows_scored)} julgamentos"

    html = _VIZ.build_dashboard(
        embed_report,
        extra_figs=judge_figs,
        extra_header_html=extra_header,
        extra_meta=extra_meta,
    )
    out_html = DATA / "report.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard unificado salvo em {out_html}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-hash", default=None,
                   help="filtra judgments por config_hash (default: usa todos)")
    p.add_argument("--no-html", action="store_true",
                   help="só gera judge_report.json, não reescreve report.html")
    main(p.parse_args())
