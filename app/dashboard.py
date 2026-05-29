"""
NeuroGuard — Interactive Safety-Analysis Dashboard
===================================================
Run:  streamlit run app/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
DATA = ROOT / "data" / "processed"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data loading (cached) ────────────────────────────────────────────────
@st.cache_data
def load_manifest():
    with open(RESULTS / "manifest.json") as f:
        return json.load(f)


@st.cache_data
def load_metrics(experiment: str):
    path = RESULTS / experiment / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_sessions():
    d2a = pd.read_parquet(DATA / "d2a_sessions.parquet")
    d2c = pd.read_parquet(DATA / "d2c_labels.parquet")
    df = d2a.merge(
        d2c[["session_id", "overall_unsafe"]], on="session_id", suffixes=("", "_label")
    )
    unsafe_col = next(c for c in df.columns if c.startswith("overall_unsafe"))
    if unsafe_col != "overall_unsafe":
        df.rename(columns={unsafe_col: "overall_unsafe"}, inplace=True)
    return df


manifest = load_manifest()
df = load_sessions()

# ── Sidebar navigation ───────────────────────────────────────────────────
st.sidebar.title("🛡️ NeuroGuard")
st.sidebar.caption("AI Safety Behaviour Analysis")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Explorer",
        "Model Performance",
        "Interpretability",
        "Bayesian Analysis",
        "Markov Chains",
        "Figure Gallery",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**{len(manifest['figures'])}** figures · "
    f"**{len(manifest['experiments'])}** experiments"
)

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("NeuroGuard — AI Safety Behaviour Dashboard")
    st.markdown(
        "Interactive exploration of how AI systems behave under adversarial "
        "pressure in safety-critical medical scenarios."
    )

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    n_sessions = len(df)
    n_unsafe = int(df["overall_unsafe"].sum())
    unsafe_rate = n_unsafe / n_sessions

    col1.metric("Sessions", n_sessions)
    col2.metric("Unsafe", n_unsafe, f"{unsafe_rate:.1%}")
    col3.metric("Scenarios", df["scenario_id"].nunique())
    col4.metric("Pressure Conditions", df["pressure_id"].nunique())

    st.markdown("---")

    # Quick summaries
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Unsafe Rate by Pressure")
        piv = (
            df.groupby("pressure_id")["overall_unsafe"]
            .agg(["sum", "count"])
            .assign(rate=lambda x: x["sum"] / x["count"])
            .sort_values("rate", ascending=False)
            .reset_index()
        )
        fig = px.bar(
            piv,
            x="pressure_id",
            y="rate",
            text=piv["rate"].map("{:.1%}".format),
            color="rate",
            color_continuous_scale="OrRd",
            labels={"pressure_id": "Pressure", "rate": "Unsafe Rate"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("Unsafe Rate by Scenario")
        piv = (
            df.groupby("scenario_id")["overall_unsafe"]
            .agg(["sum", "count"])
            .assign(rate=lambda x: x["sum"] / x["count"])
            .sort_values("rate", ascending=False)
            .reset_index()
        )
        fig = px.bar(
            piv,
            x="scenario_id",
            y="rate",
            text=piv["rate"].map("{:.1%}".format),
            color="rate",
            color_continuous_scale="OrRd",
            labels={"scenario_id": "Scenario", "rate": "Unsafe Rate"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, width="stretch")

    # Experiment timeline
    st.subheader("Experiment Timeline")
    timeline_data = []
    for name, exp in manifest["experiments"].items():
        if "timestamp" in exp:
            timeline_data.append(
                {
                    "Experiment": name,
                    "Timestamp": exp["timestamp"],
                    "Description": exp.get("description", ""),
                }
            )
    if timeline_data:
        tl_df = pd.DataFrame(timeline_data)
        tl_df["Timestamp"] = pd.to_datetime(tl_df["Timestamp"])
        st.dataframe(tl_df.sort_values("Timestamp"), width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Data Explorer
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.title("Data Explorer")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        scenarios = st.multiselect(
            "Scenario",
            sorted(df["scenario_id"].unique()),
            default=sorted(df["scenario_id"].unique()),
        )
    with col2:
        pressures = st.multiselect(
            "Pressure",
            sorted(df["pressure_id"].unique()),
            default=sorted(df["pressure_id"].unique()),
        )
    with col3:
        monitoring = st.multiselect(
            "Monitoring",
            sorted(df["monitoring_id"].unique()),
            default=sorted(df["monitoring_id"].unique()),
        )

    filtered = df[
        df["scenario_id"].isin(scenarios)
        & df["pressure_id"].isin(pressures)
        & df["monitoring_id"].isin(monitoring)
    ]

    st.metric(
        "Filtered Sessions",
        len(filtered),
        f"{filtered['overall_unsafe'].mean():.1%} unsafe",
    )

    # Heatmap
    st.subheader("Unsafe Rate: Scenario x Pressure")
    cross = (
        filtered.groupby(["scenario_id", "pressure_id"])["overall_unsafe"]
        .mean()
        .reset_index()
    )
    hm = cross.pivot(
        index="scenario_id", columns="pressure_id", values="overall_unsafe"
    )
    fig = px.imshow(
        hm,
        text_auto=".0%",
        color_continuous_scale="RdYlGn_r",
        labels={"color": "Unsafe Rate"},
        aspect="auto",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

    # Monitoring effect
    st.subheader("Monitoring Effect")
    mon_piv = (
        filtered.groupby(["monitoring_id", "scenario_id"])["overall_unsafe"]
        .mean()
        .reset_index()
    )
    fig = px.bar(
        mon_piv,
        x="scenario_id",
        y="overall_unsafe",
        color="monitoring_id",
        barmode="group",
        labels={"overall_unsafe": "Unsafe Rate", "scenario_id": "Scenario"},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

    # Raw data
    with st.expander("Raw session data"):
        st.dataframe(filtered, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Model Performance
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance Comparison")

    metrics = load_metrics("classification")
    if metrics is None:
        st.error("Classification metrics not found.")
        st.stop()

    # Build comparison table from nested metrics structure
    model_data = []
    for model_name, model_info in metrics.get("models", {}).items():
        row = {"Model": model_name}
        m = model_info.get("metrics", {})
        for metric_key in ["roc_auc", "pr_auc", "f1", "mcc"]:
            if metric_key in m:
                row[f"{metric_key}_mean"] = m[metric_key].get("mean")
                row[f"{metric_key}_std"] = m[metric_key].get("std")
        model_data.append(row)

    if model_data:
        mdf = pd.DataFrame(model_data).set_index("Model")

        # Highlight best
        st.subheader("Nested Cross-Validation Results")
        mean_cols = [c for c in mdf.columns if "mean" in c]
        st.dataframe(
            mdf.style.format("{:.4f}", na_rep="—").highlight_max(
                axis=0, subset=mean_cols, props="background-color: #d4edda;"
            ),
            width="stretch",
        )

        # Interactive bar chart
        available_metrics = [c for c in mdf.columns if c.endswith("_mean")]
        metric_choice = st.selectbox("Metric", available_metrics)
        std_col = metric_choice.replace("_mean", "_std")
        fig = px.bar(
            mdf.reset_index().sort_values(metric_choice, ascending=False),
            x="Model",
            y=metric_choice,
            color=metric_choice,
            color_continuous_scale="Viridis",
            error_y=std_col if std_col in mdf.columns else None,
        )
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    # Show figures
    st.subheader("Classification Figures")
    for fig_path in manifest["experiments"]["classification"]["figures"]:
        full_path = ROOT / fig_path
        if full_path.exists():
            st.image(str(full_path), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Interpretability
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Interpretability":
    st.title("Model Interpretability")

    metrics = load_metrics("interpretability")

    if metrics:
        # Permutation importance
        st.subheader("Permutation Importance — Top Features")
        perm_data = []
        for model_name, model_info in metrics.get("permutation_importance", {}).items():
            for feat in model_info.get("top_5", []):
                perm_data.append(
                    {
                        "Model": model_name,
                        "Feature": feat["feature"],
                        "Importance": feat["importance"],
                    }
                )
        if perm_data:
            perm_df = pd.DataFrame(perm_data)
            fig = px.bar(
                perm_df,
                x="Importance",
                y="Feature",
                color="Model",
                orientation="h",
                barmode="group",
                height=400,
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

        # Feature-safety hypothesis mapping
        if "feature_safety_mapping" in metrics:
            st.subheader("Feature → Safety Hypothesis Mapping")
            mapping = metrics["feature_safety_mapping"]
            if isinstance(mapping, list):
                st.dataframe(pd.DataFrame(mapping), width="stretch", hide_index=True)
            elif isinstance(mapping, dict):
                st.json(mapping)

    # Show figures
    st.subheader("Interpretability Figures")
    for fig_path in manifest["experiments"]["interpretability"]["figures"]:
        full_path = ROOT / fig_path
        if full_path.exists():
            st.image(str(full_path), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Bayesian Analysis
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Bayesian Analysis":
    st.title("Bayesian Analysis")

    metrics = load_metrics("bayesian")
    if metrics is None:
        st.error("Bayesian metrics not found.")
        st.stop()

    # Pressure posteriors
    st.subheader("Posterior Estimates: Unsafe Rate by Pressure")
    post_data = metrics.get("pressure_posteriors", [])
    if post_data:
        post_df = pd.DataFrame(post_data)
        fig = go.Figure()
        for _, row in post_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["cri_low"], row["post_mean"], row["cri_high"]],
                    y=[row["condition"]] * 3,
                    mode="lines+markers",
                    marker=dict(size=[6, 12, 6]),
                    name=row["condition"],
                    showlegend=False,
                    hovertemplate=f"{row['condition']}<br>Mean: {row['post_mean']:.3f}<br>95% CrI: [{row['cri_low']:.3f}, {row['cri_high']:.3f}]",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[row["mle"]],
                    y=[row["condition"]],
                    mode="markers",
                    marker=dict(symbol="x", size=10, color="red"),
                    showlegend=False,
                )
            )
        fig.update_layout(
            height=350,
            xaxis_title="Unsafe Rate (θ)",
            title="Forest Plot: Posterior Mean & 95% Credible Intervals",
        )
        st.plotly_chart(fig, width="stretch")

    # Pressure contrasts
    st.subheader("Bayesian Pressure Contrasts (vs Baseline)")
    contrasts = metrics.get("pressure_contrasts", {})
    if contrasts:
        c_data = []
        for pressure, info in contrasts.items():
            c_data.append(
                {
                    "Pressure": pressure,
                    "P(θ > baseline)": info["prob_greater_than_baseline"],
                    "Mean Δ": info["mean_delta"],
                    "95% CrI Lower": info["cri_delta"][0],
                    "95% CrI Upper": info["cri_delta"][1],
                    "Evidence": info["evidence"],
                }
            )
        st.dataframe(pd.DataFrame(c_data), width="stretch", hide_index=True)

    # Monitoring
    st.subheader("Monitoring Effect")
    mon = metrics.get("monitoring_effect", {})
    if mon:
        c1, c2, c3 = st.columns(3)
        c1.metric("P(monitored safer)", f"{mon['prob_monitored_safer']:.3f}")
        c2.metric("Mean Δ", f"{mon['mean_delta']:+.4f}")
        c3.metric("Conclusion", mon["conclusion"])

    # MCMC diagnostics
    mcmc = metrics.get("methods", {}).get("mcmc", {})
    if mcmc:
        st.subheader("MCMC Diagnostics")
        d1, d2 = st.columns(2)
        d1.metric("Max R-hat", f"{mcmc.get('max_rhat', 'N/A')}")
        d2.metric("Min ESS (bulk)", f"{mcmc.get('min_ess_bulk', 'N/A'):.0f}")

    # Figures
    st.subheader("Bayesian Figures")
    for fig_path in manifest["experiments"]["bayesian"]["figures"]:
        full_path = ROOT / fig_path
        if full_path.exists():
            st.image(str(full_path), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Markov Chains
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Markov Chains":
    st.title("Markov Chain — Sequential Behaviour Analysis")

    metrics = load_metrics("markov")
    if metrics is None:
        st.error("Markov metrics not found.")
        st.stop()

    # State distribution
    st.subheader("Behavioural State Distribution")
    states = metrics.get("overall_stats", {}).get("state_distribution", {})
    if states:
        fig = px.pie(
            names=list(states.keys()),
            values=list(states.values()),
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    # Safe vs unsafe stationary distributions
    st.subheader("Stationary Distributions: Safe vs Unsafe")
    safe_unsafe = metrics.get("safe_vs_unsafe", {})
    if safe_unsafe:
        safe_pi = safe_unsafe.get("safe_stationary", {})
        unsafe_pi = safe_unsafe.get("unsafe_stationary", {})
        state_names = list(safe_pi.keys())

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Safe",
                x=state_names,
                y=[safe_pi[s] for s in state_names],
                marker_color="#3498db",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Unsafe",
                x=state_names,
                y=[unsafe_pi[s] for s in state_names],
                marker_color="#e74c3c",
            )
        )
        fig.update_layout(
            barmode="group", height=350, yaxis_title="Stationary Probability"
        )
        st.plotly_chart(fig, width="stretch")

    # Mean first passage
    mfp = safe_unsafe.get("mean_first_passage_claiming_to_citing", {})
    if mfp:
        st.subheader("Grounding Discipline: Claiming → Citing")
        c1, c2, c3 = st.columns(3)
        c1.metric("Safe sessions", f"{mfp['safe']:.1f} sentences")
        c2.metric("Unsafe sessions", f"{mfp['unsafe']:.1f} sentences")
        c3.metric("Difference", f"{mfp['unsafe'] - mfp['safe']:+.1f} sentences")

    # Figures
    st.subheader("Markov Figures")
    for fig_path in manifest["experiments"]["markov"]["figures"]:
        full_path = ROOT / fig_path
        if full_path.exists():
            st.image(str(full_path), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Figure Gallery
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Figure Gallery":
    st.title("Figure Gallery")

    # Filter by experiment
    experiments = ["All", *list(manifest["experiments"].keys())]
    exp_filter = st.selectbox("Filter by experiment", experiments)

    figures = manifest["figures"]
    if exp_filter != "All":
        figures = [f for f in figures if f["experiment"] == exp_filter]

    st.markdown(f"**{len(figures)} figures**")

    # Display in a grid
    cols_per_row = st.slider("Columns", 1, 4, 2)
    for i in range(0, len(figures), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(figures):
                fig_info = figures[idx]
                full_path = ROOT / fig_info["file"]
                if full_path.exists():
                    with col:
                        caption = f"Fig {fig_info['number']:02d}: {fig_info['name'].replace('_', ' ').title()}"
                        st.image(str(full_path), caption=caption, width="stretch")
