#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 12:53:58 2026

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
frontier_files = sorted(glob.glob("frontier_*.parquet"))
if not frontier_files:
    st.error("No frontier_*.parquet files found in this folder.")
    st.stop()

groups = [os.path.basename(f).replace("frontier_", "").replace(".parquet", "") for f in frontier_files]
g = st.sidebar.selectbox("Group", groups)

# View toggle state
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Frontier"

# ----------------------------
# Load frontier data
# ----------------------------
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
# Helpers
# ----------------------------
def weights_df_from_row(row):
    w = pd.DataFrame({"ticker": ticker_cols, "weight": row[ticker_cols].astype(float).values})

    w.loc[w["weight"].abs() < 1e-12, "weight"] = 0.0
    w.loc[w["weight"] < 0, "weight"] = 0.0

    s = w["weight"].sum()
    if s > 0:
        w["weight"] = w["weight"] / s

    w = w.sort_values("weight", ascending=False).reset_index(drop=True)
    w["cum_weight"] = w["weight"].cumsum()

    w_disp = w.copy()
    w_disp["weight"] = w_disp["weight"].map(lambda x: f"{x:.2%}")
    w_disp["cum_weight"] = w_disp["cum_weight"].map(lambda x: f"{x:.2%}")

    return w, w_disp

def table_height_for(n_rows: int):
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

    wdf, wdf_disp = weights_df_from_row(row)
    h = table_height_for(len(wdf_disp))

    st.dataframe(
        wdf_disp,
        use_container_width=True,
        height=h,
        hide_index=True,
    )
    st.caption(f"Weight sum: {float(wdf['weight'].sum()):.4f}")

def base100_from_prices(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Buy & hold Base 100:
      Base100(t) = 100 * sum_i w_i * (P_i(t)/P_i(0))
    """
    prices = prices.copy()
    prices = prices.loc[:, weights.index]  # ensure same columns order
    rel = prices / prices.iloc[0]
    b100 = 100.0 * rel.mul(weights, axis=1).sum(axis=1)
    return b100

# ----------------------------
# Layout: top row (chart + selected), bottom row (GMV + Max Return)
# ----------------------------
top_left, top_right = st.columns([2.2, 1.0], gap="large")

# Buttons to switch view (no scrolling)
with top_left:
    b1, b2 = st.columns([1, 1])
    if b1.button("Frontier view", use_container_width=True):
        st.session_state.view_mode = "Frontier"
    if b2.button("Base 100 view", use_container_width=True):
        st.session_state.view_mode = "Base100"

# ---- Frontier figure (for Frontier view)
frontier_fig = go.Figure()

frontier_fig.add_trace(
    go.Scatter(
        x=df_frontier["vol"],
        y=df_frontier["ret"],
        mode="lines+markers",
        name="Frontier",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    )
)
frontier_fig.add_trace(
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
frontier_fig.add_trace(
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

frontier_fig.update_layout(
    title=f"Efficient Frontier — {g}",
    xaxis_title="Volatility (annualized)",
    yaxis_title="Expected Return (annualized)",
    legend_title="",
    margin=dict(l=10, r=10, t=50, b=10),
)
frontier_fig.update_xaxes(tickformat=".1%")
frontier_fig.update_yaxes(tickformat=".1%")

# ---- Determine selected point (persist across view switches)
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = int(df_frontier["vol"].idxmin())

# Render either Frontier or Base100 in the left panel
with top_left:
    if st.session_state.view_mode == "Frontier":
        event = st.plotly_chart(
            frontier_fig,
            key=f"frontier_{g}",
            on_select="rerun",
            selection_mode="points",
            height=520,
            use_container_width=True,
        )

        # Only accept clicks on frontier trace (curve_number == 0)
        if event and event.selection and event.selection.get("points"):
            p0 = event.selection["points"][0]
            if p0.get("curve_number", None) == 0:
                st.session_state.selected_idx = int(p0["point_index"])

    else:
        # Base 100 view
        price_path = f"prices_{g}.parquet"
        if not os.path.exists(price_path):
            st.error(f"Missing {price_path}. Export it from the notebook (prices_<group>.parquet).")
        else:
            prices = pd.read_parquet(price_path).copy()
            if "date" in prices.columns:
                prices["date"] = pd.to_datetime(prices["date"])
                prices = prices.set_index("date")

            # Build weight vectors
            row_sel = df_frontier.iloc[int(st.session_state.selected_idx)]
            w_sel = weights_df_from_row(row_sel)[0].set_index("ticker")["weight"]
            w_gmv = weights_df_from_row(row_gmv)[0].set_index("ticker")["weight"]
            w_max = weights_df_from_row(row_max)[0].set_index("ticker")["weight"]


            # Compute Base100
            b_sel = base100_from_prices(prices, w_sel)
            b_gmv = base100_from_prices(prices, w_gmv)
            b_max = base100_from_prices(prices, w_max)

            # Controls
            c1, c2 = st.columns(2)
            show_gmv = c1.checkbox("Show GMV", value=True)
            show_max = c2.checkbox("Show Max Return", value=True)

            base_fig = go.Figure()
            base_fig.add_trace(go.Scatter(x=b_sel.index, y=b_sel.values, mode="lines", name="Selected"))
            if show_gmv:
                base_fig.add_trace(go.Scatter(x=b_gmv.index, y=b_gmv.values, mode="lines", name="GMV"))
            if show_max:
                base_fig.add_trace(go.Scatter(x=b_max.index, y=b_max.values, mode="lines", name="Max Return"))

            base_fig.update_layout(
                title=f"Base 100 (Buy & Hold) — {g}",
                xaxis_title="Date",
                yaxis_title="Index (Base 100)",
                margin=dict(l=10, r=10, t=50, b=10),
            )

            st.plotly_chart(base_fig, height=520, use_container_width=True)
            st.caption("Base 100 assumes buy & hold with initial weights (no rebalancing).")

# Selected row for the right panel (weights/stats)
row_sel = df_frontier.iloc[int(st.session_state.selected_idx)]

with top_right:
    render_point(f"Selected point — #{int(st.session_state.selected_idx)}", row_sel)

st.divider()

bottom_left, bottom_right = st.columns(2, gap="large")
with bottom_left:
    render_point("GMV (minimum variance)", row_gmv)

with bottom_right:
    render_point("Max Return (long-only)", row_max)
