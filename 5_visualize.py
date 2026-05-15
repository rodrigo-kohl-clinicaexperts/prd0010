import json
import argparse
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA = Path(__file__).parent / "data"


def load_report():
    path = DATA / "report.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode 4_report.py primeiro.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Painéis individuais — cada um devolve uma lista de traces + layout hints
# --------------------------------------------------------------------------

def fig_means_by_metric(report):
    """Barras agrupadas: para cada métrica, uma barra por modelo."""
    means = report.get("means_by_metric", {})
    if not means:
        return None
    metrics = list(means.keys())
    # união de todos os modelos que aparecem
    models = sorted({m for d in means.values() for m in d})

    fig = go.Figure()
    for model in models:
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=[means[metric].get(model) for metric in metrics],
        ))
    fig.update_layout(
        title="Média por modelo × métrica (maior = mais próximo da referência)",
        barmode="group",
        xaxis_title="métrica",
        yaxis_title="similaridade média",
        legend_title="modelo",
    )
    return fig


def fig_spearman_heatmap(report):
    """Heatmap da concordância de Spearman entre as métricas."""
    pares = report.get("metric_agreement_spearman", [])
    if not pares:
        return None
    # monta a matriz simétrica
    metrics = sorted({p["metric_a"] for p in pares} | {p["metric_b"] for p in pares})
    idx = {m: i for i, m in enumerate(metrics)}
    n = len(metrics)
    matrix = [[None] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0  # diagonal: métrica com ela mesma
    for p in pares:
        i, j = idx[p["metric_a"]], idx[p["metric_b"]]
        matrix[i][j] = p["spearman"]
        matrix[j][i] = p["spearman"]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=metrics,
        y=metrics,
        zmin=-1, zmax=1,
        colorscale="RdYlGn",  # vermelho = discorda, verde = concorda
        text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in matrix],
        texttemplate="%{text}",
        colorbar=dict(title="Spearman"),
    ))
    fig.update_layout(
        title="Concordância entre métricas (Spearman entre rankings) — "
              "verde = robusto, vermelho = frágil",
    )
    return fig


def fig_breakdown_heatmap(report):
    """Heatmap modelo × question_type."""
    breakdown = report.get("breakdown_by_question_type", {})
    if not breakdown:
        return None
    qtypes = list(breakdown.keys())
    models = sorted({m for d in breakdown.values() for m in d})

    # z[modelo][tipo]
    z = [[breakdown[qt].get(model) for qt in qtypes] for model in models]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=qtypes,
        y=models,
        colorscale="Viridis",
        text=[[f"{v:.3f}" if v is not None else "" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title="similaridade"),
    ))
    fig.update_layout(
        title="Desempenho por question_type (métrica primária)",
        xaxis_title="question_type",
        yaxis_title="modelo",
    )
    return fig


def fig_completeness(report):
    """Barras de completude — n por modelo, com destaque se divergem."""
    counts = report.get("n_by_model", {})
    if not counts:
        return None
    models = sorted(counts)
    vals = [counts[m] for m in models]

    # destaque: se o menor < 90% do maior, pinta as barras de alerta
    n_min, n_max = min(vals), max(vals)
    diverge = len(vals) > 1 and n_min < 0.9 * n_max
    cor = "crimson" if diverge else "steelblue"

    fig = go.Figure(data=go.Bar(
        x=models, y=vals,
        marker_color=cor,
        text=vals, textposition="outside",
    ))
    titulo = "Completude — n de questões pontuadas por modelo"
    if diverge:
        titulo += "  ⚠ n DIVERGEM — médias não são diretamente comparáveis"
    fig.update_layout(
        title=titulo,
        xaxis_title="modelo",
        yaxis_title="n questões",
    )
    return fig


# --------------------------------------------------------------------------

def build_dashboard(report):
    """
    Monta um HTML único com todos os painéis empilhados. Cada figura vira uma
    div; concatenamos no mesmo arquivo. Plotly embute o JS na primeira figura
    (include_plotlyjs) e as demais reusam.
    """
    figs = []
    for builder in (fig_means_by_metric, fig_spearman_heatmap,
                    fig_breakdown_heatmap, fig_completeness):
        f = builder(report)
        if f is not None:
            figs.append(f)

    if not figs:
        raise RuntimeError("report.json não tem dados suficientes para nenhum "
                           "gráfico — confira se 4_report.py rodou direito.")

    partes = []
    for i, f in enumerate(figs):
        # só a primeira figura embute a lib plotly.js; as outras reaproveitam
        partes.append(f.to_html(
            full_html=False,
            include_plotlyjs=(True if i == 0 else False),
        ))

    n_rows = report.get("n_rows", "?")
    metrics = ", ".join(report.get("metrics", []))
    html = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <title>MedPT — Benchmark de modelos médicos</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1100px; margin: 0 auto;
            padding: 20px; background: #fafafa; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    .meta {{ color: #666; margin-bottom: 24px; }}
    .panel {{ background: white; border: 1px solid #ddd; border-radius: 6px;
              margin-bottom: 20px; padding: 10px; }}
    .nota {{ background: #fff8e1; border-left: 4px solid #fbc02d;
             padding: 12px; margin-top: 24px; font-size: 0.92em; }}
  </style>
</head>
<body>
  <h1>MedPT — Benchmark de modelos médicos (PT-BR)</h1>
  <div class="meta">
    {n_rows} linhas pontuadas &middot; métricas: {metrics}
  </div>
  {"".join(f'<div class="panel">{p}</div>' for p in partes)}
  <div class="nota">
    <strong>Lembrete metodológico:</strong> estas métricas medem proximidade
    à resposta de referência do MedPT, não qualidade clínica absoluta. O MedPT
    é Q&amp;A de fórum — a referência é uma resposta humana entre várias
    possíveis. Para conclusões fortes, valide o ranking com o LLM-as-judge
    (6_llm_judge.py) num subconjunto.
  </div>
</body>
</html>"""
    return html


def main(args):
    report = load_report()
    print(f"report.json carregado: {report.get('n_rows', '?')} linhas, "
          f"métricas {report.get('metrics', [])}")

    html = build_dashboard(report)

    out_path = DATA / "report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard salvo: {out_path}")
    print("Abra no navegador para ver os gráficos interativos.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    main(p.parse_args())