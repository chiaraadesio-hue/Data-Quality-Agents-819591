"""
Streamlit dashboard for the Multi-Agent Data Quality system.

Run with:
    streamlit run app.py

The app accepts a CSV upload (or one of the bundled sample datasets),
runs the deterministic pipeline (no Gemini call required, so the app
works without an API key), and shows the results across five tabs:

    1. Overview        - reliability score, top-line metrics
    2. Schema          - column-level diagnoses, duplicate-column pairs
    3. Completeness    - bar chart of completeness per column
    4. Consistency     - format issues, cross-column violations, dups
    5. Anomalies       - outliers and rare categories
    6. Auto-Fix        - download cleaned CSV + audit log

If a GOOGLE_API_KEY is configured (via .env or Streamlit secrets), an
extra "AI Insights" panel calls Gemini for narrative remediation text;
otherwise the panel is hidden so the app remains usable offline.
"""
from __future__ import annotations

import io
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import dq_agents as dq

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NoiPA Data Quality Agents",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand colors (kept in sync with the mid-check slide deck).
BRAND = {
    "primary":   "#5D2E8C",   # deep purple
    "secondary": "#D14591",   # magenta
    "accent":    "#7F77DD",   # lavender
    "success":   "#1FB964",
    "warning":   "#F5C166",
    "danger":    "#E24B4A",
    "muted":     "#85B7EB",
}

st.markdown(
    f"""
    <style>
        .main {{ padding-top: 1rem; }}
        h1, h2, h3 {{ color: {BRAND['primary']}; }}
        .stMetric {{
            background: linear-gradient(135deg, #faf7ff 0%, #f3edff 100%);
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid {BRAND['secondary']};
        }}
        .score-banner {{
            background: linear-gradient(135deg, {BRAND['primary']} 0%, {BRAND['secondary']} 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 16px;
        }}
        .score-banner h1 {{ color: white !important; margin: 0; font-size: 3rem; }}
        .score-banner p  {{ color: rgba(255,255,255,0.9); margin: 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar: dataset selection
# ---------------------------------------------------------------------------
st.sidebar.title(" Data Quality Agents")
st.sidebar.markdown("**Reply x LUISS — ML 2025/26**")
st.sidebar.markdown("---")

st.sidebar.header("1. Choose a dataset")
sample_paths = {
    "spesa.csv (Reply, 7.5k rows)":              "data/spesa.csv",
    "attivazioniCessazioni.csv (Reply, 20k rows)": "data/attivazioniCessazioni.csv",
    "dataset_noipa_1.csv (synthetic, 51 rows)":  "data/dataset_noipa_1.csv",
    "dataset_noipa_2.csv (synthetic, 40 rows)":  "data/dataset_noipa_2.csv",
}
available_samples = {k: v for k, v in sample_paths.items() if os.path.exists(v)}

mode = st.sidebar.radio(
    "Source",
    options=["Sample dataset", "Upload CSV"],
    index=0,
)

uploaded_file = None
sample_choice = None
if mode == "Sample dataset" and available_samples:
    sample_choice = st.sidebar.selectbox(
        "Sample", list(available_samples.keys())
    )
elif mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "CSV file", type=["csv"]
    )
else:
    st.sidebar.warning("No sample datasets found in data/.")

st.sidebar.markdown("---")
st.sidebar.header("2. About")
st.sidebar.info(
    "Six specialized agents validate, score, and clean a NoiPA dataset. "
    "Pipeline: Schema → Completeness → Consistency → Anomaly → Auto-Fix → Remediation."
)


# ---------------------------------------------------------------------------
# Load the chosen dataset
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer, key: str) -> pd.DataFrame:
    """Cached CSV reader. `key` invalidates the cache when the source changes."""
    return pd.read_csv(path_or_buffer, dtype=str)


df: pd.DataFrame | None = None
dataset_label = ""
if uploaded_file is not None:
    df = load_csv(uploaded_file, key=uploaded_file.name)
    dataset_label = uploaded_file.name
elif sample_choice:
    df = load_csv(available_samples[sample_choice], key=available_samples[sample_choice])
    dataset_label = sample_choice

if df is None:
    st.title(" NoiPA Data Quality Agents")
    st.markdown(
        """
        ### Welcome
        This dashboard runs a **multi-agent data-quality pipeline** built with LangGraph
        and Google Gemini against a NoiPA dataset.

        Pick a sample dataset on the left, or upload your own CSV, to get started.
        """
    )
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Logo_NoiPA.png/320px-Logo_NoiPA.png",
        width=240,
    )
    st.stop()


# ---------------------------------------------------------------------------
# Run the pipeline (cached so we don't re-run on every widget interaction)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running 6 agents on the dataset...")
def run_pipeline(df_serialized: str) -> dict:
    """Cache key is the CSV string; results are returned as a dict."""
    df_in = pd.read_csv(io.StringIO(df_serialized), dtype=str)
    state = dq.run_pipeline_local(df_in)
    state["cleaned_df_csv"] = state["cleaned_df"].to_csv(index=False)
    state["dataframe_csv"]  = df_in.to_csv(index=False)
    # Re-score after the auto-fix to compute the score lift.
    post = dq.run_pipeline_local(state["cleaned_df"])
    state["post_scores"] = post["scores"]
    # Don't put DataFrames in the cache return value (not picklable safely).
    state.pop("dataframe", None)
    state.pop("cleaned_df", None)
    return state


csv_serialized = df.to_csv(index=False)
state = run_pipeline(csv_serialized)
df_clean = pd.read_csv(io.StringIO(state["cleaned_df_csv"]), dtype=str)


# ---------------------------------------------------------------------------
# Header + reliability score banner
# ---------------------------------------------------------------------------
score = state["scores"]["overall"]
score_class = (
    "🟢 Excellent" if score >= 90 else
    "🟡 Good"      if score >= 75 else
    "🟠 Fair"      if score >= 60 else
    "🔴 Poor"
)

st.title(" NoiPA Data Quality Audit")
st.caption(f"Dataset: **{dataset_label}** · {df.shape[0]:,} rows × {df.shape[1]} columns")

st.markdown(
    f"""
    <div class="score-banner">
        <p style="font-size:0.9rem;margin-bottom:4px;">RELIABILITY SCORE</p>
        <h1>{score:.1f}<span style="font-size:1.4rem;opacity:0.7;">/100</span></h1>
        <p style="margin-top:4px;font-weight:600;">{score_class}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Completeness (40%)", f"{state['scores']['completeness']:.1f}")
c2.metric("Schema (30%)",       f"{state['scores']['schema']:.1f}")
c3.metric("Consistency (20%)",  f"{state['scores']['consistency']:.1f}")
c4.metric("Anomaly (10%)",      f"{state['scores']['anomaly']:.1f}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_schema, tab_comp, tab_cons, tab_anom, tab_fix = st.tabs([
    " Overview",
    " Schema",
    " Completeness",
    " Consistency",
    " Anomalies",
    " Auto-Fix",
])


# ============================ OVERVIEW ====================================
with tab_overview:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Score breakdown")
        score_fig = go.Figure()
        labels = ["Completeness<br>(40%)", "Schema<br>(30%)", "Consistency<br>(20%)",
                  "Anomaly<br>(10%)", "OVERALL"]
        values = [
            state['scores']['completeness'],
            state['scores']['schema'],
            state['scores']['consistency'],
            state['scores']['anomaly'],
            state['scores']['overall'],
        ]
        colors = [BRAND['success'], BRAND['muted'], BRAND['warning'],
                  BRAND['secondary'], BRAND['accent']]
        score_fig.add_bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
        )
        score_fig.update_layout(
            yaxis=dict(range=[0, 110], title="Score (0-100)"),
            showlegend=False,
            height=380,
            margin=dict(t=20, b=20, l=40, r=20),
        )
        st.plotly_chart(score_fig, use_container_width=True)

    with col_right:
        st.subheader("Issue summary")
        summary_rows = [
            ("Naming issues",       state['schema']['summary']['n_naming_issues']),
            ("Type issues",         state['schema']['summary']['n_type_issues']),
            ("Duplicate columns",   state['schema']['summary']['n_duplicates']),
            ("Missing cells",       state['completeness']['summary']['n_missing_cells']),
            ("Sparse columns",      state['completeness']['summary']['n_sparse_columns']),
            ("Format issues",       state['consistency']['summary']['n_format_issues']),
            ("Cross-column issues", state['consistency']['summary']['n_cross_column']),
            ("Exact dup. groups",   state['consistency']['summary']['n_exact_dup_groups']),
            ("Numerical outliers",  state['anomaly']['summary']['n_outliers']),
            ("Rare categories",     state['anomaly']['summary']['n_rare_categories']),
        ]
        sdf = pd.DataFrame(summary_rows, columns=["Category", "Count"])
        sdf["Severity"] = sdf["Count"].apply(
            lambda x: "🔴 HIGH" if x > 50 else "🟠 MEDIUM" if x > 5 else "🟡 LOW" if x > 0 else "🟢 OK"
        )
        st.dataframe(sdf, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Score lift after auto-fix")
    post = state["post_scores"]
    cats = ["Completeness", "Schema", "Consistency", "Anomaly", "Overall"]
    before_v = [state['scores']['completeness'], state['scores']['schema'],
                state['scores']['consistency'], state['scores']['anomaly'],
                state['scores']['overall']]
    after_v = [post['completeness'], post['schema'],
               post['consistency'], post['anomaly'], post['overall']]

    cmp_fig = go.Figure()
    cmp_fig.add_bar(name="Before", x=cats, y=before_v, marker_color=BRAND['warning'],
                    text=[f"{v:.0f}" for v in before_v], textposition="outside")
    cmp_fig.add_bar(name="After",  x=cats, y=after_v,  marker_color=BRAND['success'],
                    text=[f"{v:.0f}" for v in after_v], textposition="outside")
    cmp_fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 115], title="Score (0-100)"),
        height=360,
        margin=dict(t=20, b=20, l=40, r=20),
    )
    st.plotly_chart(cmp_fig, use_container_width=True)
    st.caption(
        f"After applying the {len(state['fix_log'])} safe fixes the score "
        f"moves from **{state['scores']['overall']:.1f}** to **{post['overall']:.1f}** "
        f"(+{post['overall'] - state['scores']['overall']:.1f} points)."
    )


# ============================ SCHEMA ====================================
with tab_schema:
    st.subheader("Column-level schema diagnosis")
    rows = []
    for col, info in state['schema']['columns'].items():
        rows.append({
            "Column": col,
            "Status": "✅ OK" if info['status'] == "OK" else "❌ FAIL",
            "Naming severity": info['naming']['severity'],
            "Issues": ", ".join(info['naming']['issues']) or "—",
            "Suggested name": info['naming']['suggestion'],
            "Expected type": info['expected_type'],
            "Type issues": "; ".join(info['type_issues']) or "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if state['schema']['duplicate_columns']:
        st.subheader("Potential duplicate columns")
        st.caption("Pairs whose values match in ≥95% of the rows. Pairs ≥0.97 are auto-dropped.")
        dup_rows = [
            {"Column A": a, "Column B": b, "Similarity": f"{sim:.2%}",
             "Auto-drop": "✅" if sim >= 0.97 else "—"}
            for a, b, sim in state['schema']['duplicate_columns']
        ]
        st.dataframe(pd.DataFrame(dup_rows), hide_index=True, use_container_width=True)
    else:
        st.success("No duplicate columns detected.")


# ============================ COMPLETENESS ==============================
with tab_comp:
    st.subheader(f"Overall completeness: {state['completeness']['overall_completeness_pct']:.2f}%")
    per_col = state['completeness']['per_column']
    cdf = pd.DataFrame([
        {
            "Column":            col,
            "Complete (%)":      info['completeness_pct'],
            "Missing":           info['total_missing'],
            "Nulls":             info['null_count'],
            "Placeholders":      info['placeholder_count'],
            "Placeholder values": ", ".join(info['placeholder_values'][:5]),
            "Sparse":            "🔴" if info['is_sparse'] else "—",
        }
        for col, info in per_col.items()
    ])

    comp_fig = px.bar(
        cdf.sort_values("Complete (%)"),
        y="Column", x="Complete (%)",
        color="Complete (%)",
        color_continuous_scale=[(0, BRAND['danger']), (0.5, BRAND['warning']), (1, BRAND['success'])],
        range_color=[0, 100],
        orientation="h",
        height=max(380, 28 * len(cdf)),
    )
    comp_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(comp_fig, use_container_width=True)

    st.dataframe(cdf, hide_index=True, use_container_width=True)


# ============================ CONSISTENCY ===============================
with tab_cons:
    cs = state['consistency']

    st.subheader(f"Format issues ({cs['summary']['n_format_issues']})")
    if cs['format_issues']:
        st.dataframe(
            pd.DataFrame(cs['format_issues'])[
                ['column', 'dominant', 'dominant_pct', 'minority_count', 'suggestion']
            ],
            hide_index=True, use_container_width=True,
        )
    else:
        st.success("All columns use a consistent format.")

    st.subheader(f"Cross-column violations ({cs['summary']['n_cross_column']})")
    if cs['cross_column_violations']:
        rows = [
            {"Row": v['row'], "Type": v['type'], "Detail": v['detail'][:120]}
            for v in cs['cross_column_violations'][:50]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if len(cs['cross_column_violations']) > 50:
            st.caption(f"Showing 50 of {len(cs['cross_column_violations'])} violations.")
    else:
        st.success("No cross-column logical violations.")

    st.subheader(
        f"Duplicates: {cs['summary']['n_exact_dup_groups']} exact group(s), "
        f"{cs['summary']['n_near_dup_pairs']} near-pair(s)"
    )
    if cs['duplicates']['exact']['groups']:
        st.markdown("**Exact duplicate groups (first 10)**")
        for grp in cs['duplicates']['exact']['groups'][:10]:
            st.write(f"  rows {grp['rows']}  (n={grp['n']})")


# ============================ ANOMALIES =================================
with tab_anom:
    a = state['anomaly']
    st.subheader(f"Numerical outliers ({a['summary']['n_outliers']})")
    if a['outliers']:
        adf = pd.DataFrame(a['outliers'])
        adf['iqr_low']  = adf['bounds'].apply(lambda b: b['iqr_low'])
        adf['iqr_high'] = adf['bounds'].apply(lambda b: b['iqr_high'])
        st.dataframe(
            adf[['row', 'column', 'value', 'z_score', 'iqr_low', 'iqr_high']].head(100),
            hide_index=True, use_container_width=True,
        )

        # Plot per-column distribution with outliers highlighted.
        out_cols = sorted(adf['column'].unique())
        sel = st.selectbox("Inspect column", out_cols)
        if sel:
            full = pd.to_numeric(df[sel], errors='coerce').dropna()
            outlier_idx = adf[adf['column'] == sel]['row'].tolist()
            outlier_vals = pd.to_numeric(df.loc[outlier_idx, sel], errors='coerce').dropna()
            box = go.Figure()
            box.add_trace(go.Box(y=full, name=sel, boxpoints='outliers',
                                  marker_color=BRAND['accent']))
            box.add_trace(go.Scatter(
                y=outlier_vals, x=[sel] * len(outlier_vals),
                mode='markers', name='flagged',
                marker=dict(color=BRAND['danger'], size=8, symbol='x'),
            ))
            box.update_layout(height=380, showlegend=True,
                              margin=dict(t=20, b=20, l=40, r=20))
            st.plotly_chart(box, use_container_width=True)
    else:
        st.success("No numerical outliers above the (z>3 ∧ IQR) threshold.")

    st.subheader(f"Rare categories ({a['summary']['n_rare_categories']})")
    if a['rare_categories']:
        st.dataframe(
            pd.DataFrame(a['rare_categories']).head(100),
            hide_index=True, use_container_width=True,
        )
        if len(a['rare_categories']) > 100:
            st.caption(f"Showing 100 of {len(a['rare_categories'])} rare values.")
    else:
        st.success("No rare categorical values flagged.")


# ============================ AUTO-FIX ==================================
with tab_fix:
    st.subheader(f"Auto-fix actions ({len(state['fix_log'])})")

    if not state['fix_log']:
        st.info("No fixes were needed — the dataset already passed every safe-to-fix check.")
    else:
        # Pretty-print the fix log.
        log_rows = []
        for entry in state['fix_log']:
            log_rows.append({
                "Action": entry['action'],
                "Target": entry['target'],
                "Details": json.dumps(
                    {k: v for k, v in entry.items() if k not in {'action', 'target'}},
                    ensure_ascii=False,
                )[:250],
            })
        st.dataframe(pd.DataFrame(log_rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Cleaned dataset preview")
    st.caption(
        f"Original: {df.shape[0]:,} rows × {df.shape[1]} cols · "
        f"Cleaned: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} cols"
    )
    st.dataframe(df_clean.head(20), use_container_width=True)

    st.markdown("---")
    st.subheader("Download")
    base = os.path.splitext(dataset_label)[0].replace("/", "_")

    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "⬇ cleaned CSV",
        data=state['cleaned_df_csv'],
        file_name=f"cleaned_{base}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col2.download_button(
        "⬇ fix log (JSON)",
        data=json.dumps(state['fix_log'], ensure_ascii=False, indent=2, default=str),
        file_name=f"fix_log_{base}.json",
        mime="application/json",
        use_container_width=True,
    )
    full_report = {
        "dataset":           dataset_label,
        "rows_in":           int(df.shape[0]),
        "cols_in":           int(df.shape[1]),
        "rows_out":          int(df_clean.shape[0]),
        "cols_out":          int(df_clean.shape[1]),
        "reliability_score": state['scores']['overall'],
        "sub_scores":        state['scores'],
        "post_scores":       state['post_scores'],
        "schema":            state['schema'],
        "completeness":      state['completeness'],
        "consistency":       state['consistency'],
        "anomaly":           state['anomaly'],
        "fix_log":           state['fix_log'],
    }
    col3.download_button(
        "⬇ full report (JSON)",
        data=json.dumps(full_report, ensure_ascii=False, indent=2, default=str),
        file_name=f"data_quality_report_{base}.json",
        mime="application/json",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Multi-Agent Data Quality System · built with LangGraph + Gemini 2.0 Flash. "
    "Eleonora Cappetti · Daniela Chiezzi · Chiara De Sio — Reply x LUISS, ML 2025/26."
)
