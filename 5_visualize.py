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


def figs_breakdown_heatmaps(report):
    """Uma heatmap modelo × question_type POR métrica, com coluna TOTAL
    (média global do modelo naquela métrica, vinda de means_by_metric —
    é a ponderada correta, não a média das colunas).

    Métricas vivem em escalas diferentes, então cada heatmap tem seu próprio
    colorscale honesto. A primária aparece em primeiro lugar.
    """
    breakdowns = report.get("breakdowns_by_metric", {})
    if not breakdowns:
        return []
    means_by_metric = report.get("means_by_metric", {})
    primary = report.get("primary_metric")

    # primária primeiro, depois as demais em ordem alfabética estável
    metrics_ordered = (
        ([primary] if primary in breakdowns else []) +
        [m for m in sorted(breakdowns) if m != primary]
    )

    figs = []
    for metric in metrics_ordered:
        breakdown = breakdowns[metric]
        qtypes = list(breakdown.keys())
        models = sorted({m for d in breakdown.values() for m in d})
        totals = means_by_metric.get(metric, {})

        columns = qtypes + ["TOTAL"]
        z = [
            [breakdown[qt].get(model) for qt in qtypes] + [totals.get(model)]
            for model in models
        ]
        is_primary = (metric == primary)
        suffix = " — primária" if is_primary else ""
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=columns,
            y=models,
            colorscale="Viridis",
            text=[[f"{v:.3f}" if v is not None else "" for v in row] for row in z],
            texttemplate="%{text}",
            colorbar=dict(title=metric),
        ))
        fig.add_vline(x=len(qtypes) - 0.5, line_width=2, line_color="white")
        fig.update_layout(
            title=f"Desempenho por question_type — métrica: {metric}{suffix} "
                  "(coluna TOTAL = média global do modelo)",
            xaxis_title="question_type",
            yaxis_title="modelo",
        )
        figs.append(fig)
    return figs


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

def build_dashboard(report, extra_figs=None, extra_header_html=None,
                    extra_meta=None):
    """
    HTML único com painéis de embedding empilhados. Aceita figuras extras
    (lista de plotly.Figure) inseridas depois dos painéis de embedding —
    usado pelo 7_judge_analysis pra anexar a seção do juiz no mesmo report.
    `extra_header_html` é inserido antes das extra_figs (ex: '<h2>Juiz</h2>').
    `extra_meta` é uma linha adicional na header.
    """
    figs = []
    for builder in (fig_means_by_metric, fig_spearman_heatmap,
                    figs_breakdown_heatmaps, fig_completeness):
        result = builder(report)
        if result is None:
            continue
        if isinstance(result, list):
            figs.extend(result)
        else:
            figs.append(result)
    n_embed = len(figs)

    if extra_figs:
        figs.extend(extra_figs)

    if not figs:
        raise RuntimeError("report.json não tem dados suficientes para nenhum "
                           "gráfico — confira se 4_report.py rodou direito.")

    partes = []
    for i, f in enumerate(figs):
        partes.append(f.to_html(
            full_html=False,
            include_plotlyjs=(True if i == 0 else False),
        ))

    n_rows = report.get("n_rows", "?")
    metrics = ", ".join(report.get("metrics", []))

    # monta o corpo: painéis de embedding, depois header opcional + extras
    body_panels = "".join(
        f'<div class="panel">{partes[i]}</div>' for i in range(n_embed))
    body_extra = ""
    if extra_figs:
        body_extra = (extra_header_html or "") + "".join(
            f'<div class="panel">{partes[i]}</div>'
            for i in range(n_embed, len(figs)))

    extra_meta_line = f" &middot; {extra_meta}" if extra_meta else ""

    html = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <title>MedPT — Benchmark de modelos médicos</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1100px; margin: 0 auto;
            padding: 20px; background: #fafafa; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    h2 {{ border-bottom: 1px solid #999; padding-bottom: 6px;
          margin-top: 36px; color: #333; }}
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
    {n_rows} linhas pontuadas &middot; métricas: {metrics}{extra_meta_line}
  </div>
  <h2>Métricas de embedding (proximidade à referência)</h2>
  {body_panels}
  {body_extra}
  <div class="nota">
    <strong>Lembrete metodológico:</strong> as métricas de embedding medem
    proximidade à resposta de referência do MedPT, não qualidade clínica
    absoluta. O LLM-as-judge (quando presente) dá uma segunda opinião baseada
    em coerência clínica e cobertura.
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