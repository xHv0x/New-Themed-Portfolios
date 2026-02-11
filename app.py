#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 11:23:33 2026

@author: hernanvillanueva
"""

import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ----------------------------
# Page + compact styling
# ----------------------------
st.set_page_config(page_title="Efficient Frontier", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.0rem; padding-bottom: 1.0rem;}
      h1, h2, h3 {margin-top: 0.2rem; margin-bottom: 0.4rem;}
      div[data-testid="stMetricValue"] {font-size: 2.0rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("## Efficient Frontier — Click a point to see weights")

# ----------------------------
# Load available groups
# ----------------------------
files = sorted(glob.glob("frontier_*.parquet"))
if not files:
    st.error("No frontier_*.parquet files found in this folder.")
    st.stop()

groups = [os.path.basename(f).replace("frontier_", "").replace(".parquet", "") for f in files]
g = st.sidebar.selectbox("Group", groups)

df_all = pd.read_parquet(f"frontier_{g}.parquet").copy()

if "point_type" not in df_all.columns:
    df_all["point_type"] = "Frontier"

ticker_cols = [c for c in df_all.columns if c not in ["point_type", "vol", "ret"]]

df_frontier = df_all[df_all["point_type"] == "Frontier"].copy()
if df_frontier.empty:
    df_frontier = df_all.copy()

df_frontier = df_frontier.sort_values("vol").reset_index(drop=True)

def get_anchor(ptype: str, fallback: str):
    sub = df_all[df_all["point_type"] == ptype]
    if not sub.empty:
        return sub.iloc[0]
    if fallback == "min_vol":
        return df_frontier.iloc[int(df_frontier["vol"].idxmin())]
    if fallback == "max_ret":
        return df_frontier.iloc[int(df_frontier["ret"].idxmax())]
    return df_frontier.iloc[0]

row_gmv = get_anchor("GMV", "min_vol")
row_max = get_anchor("MaxReturn", "max_ret")

# ----------------------------
# Plot
# ----------------------------
fig = go.Figure()

# Frontier trace (trace index 0)
fig.add_trace(
    go.Scatter(
        x=df_frontier["vol"],
        y=df_frontier["ret"],
        mode="lines+markers",
        name="Frontier",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    )
)

# GMV marker
fig.add_trace(
    go.Scatter(
        x=[float(row_gmv["vol"])],
        y=[float(row_gmv["ret"])],
        mode="markers",
        name="GMV",
        marker_symbol="x",
        marker_size=12,
        hovertemplate="GMV<br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    )
)

# Max Return marker
fig.add_trace(
    go.Scatter(
        x=[float(row_max["vol"])],
        y=[float(row_max["ret"])],
        mode="markers",
        name="Max Return",
        marker_symbol="diamond",
        marker_size=10,
        hovertemplate="Max Return<br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    )
)

fig.update_layout(
    title=f"Efficient Frontier — {g}",
    xaxis_title="Volatility (annualized)",
    yaxis_title="Expected Return (annualized)",
    legend_title="",
    margin=dict(l=10, r=10, t=50, b=10),
)

fig.update_xaxes(tickformat=".1%")
fig.update_yaxes(tickformat=".1%")

# ----------------------------
# Helpers for tables
# ----------------------------
def weights_df_from_row(row):
    w = pd.DataFrame({"ticker": ticker_cols, "weight": row[ticker_cols].astype(float).values})
    w = w.sort_values("weight", ascending=False).reset_index(drop=True)
    w["cum_weight"] = w["weight"].cumsum()
    return w

def table_height_for(n_rows: int):
    # compact dataframe height; ~35px/row + header
    h = 35 * (n_rows + 1) + 20
    return max(150, min(260, h))

def point_stats(row):
    ret = float(row["ret"])
    vol = float(row["vol"])
    ratio = ret / vol if vol > 0 else np.nan
    return ret, vol, ratio

def render_point(title, row):
    st.markdown(f"### {title}")
    ret, vol, ratio = point_stats(row)

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected return", f"{ret:.2%}")
    c2.metric("Volatility", f"{vol:.2%}")
    c3.metric("Ret/Vol", f"{ratio:.2f}" if np.isfinite(ratio) else "—")

    wdf = weights_df_from_row(row)
    h = table_height_for(len(wdf))

    st.dataframe(
        wdf.style.format({"weight": "{:.2%}", "cum_weight": "{:.2%}"}),
        use_container_width=True,
        height=h,
    )
    st.caption(f"Weight sum: {float(wdf['weight'].sum()):.4f}")

# ----------------------------
# Layout: top row (chart + selected), bottom row (GMV + Max Return)
# ----------------------------
top_left, top_right = st.columns([2.2, 1.0], gap="large")

with top_left:
    event = st.plotly_chart(
        fig,
        key=f"frontier_{g}",
        on_select="rerun",
        selection_mode="points",
        height=480,  # smaller so we fit the bottom row without scrolling
        use_container_width=True,
    )

# Selection logic: only accept clicks on the frontier trace (curve_number == 0)
selected_idx = None
if event and event.selection and event.selection.get("points"):
    p0 = event.selection["points"][0]
    if p0.get("curve_number", None) == 0:
        selected_idx = int(p0["point_index"])

# Default selection: min-vol frontier point
if selected_idx is None:
    selected_idx = int(df_frontier["vol"].idxmin())

row_sel = df_frontier.iloc[selected_idx]

with top_right:
    render_point("Selected point", row_sel)

st.divider()

bottom_left, bottom_right = st.columns(2, gap="large")
with bottom_left:
    render_point("GMV (minimum variance)", row_gmv)

with bottom_right:
    render_point("Max Return (long-only)", row_max)
