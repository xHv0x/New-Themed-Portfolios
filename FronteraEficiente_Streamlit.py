#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 11:23:33 2026

@author: hernanvillanueva
"""

# ============================================================
# Classic Efficient Frontier (Long-Only) per Group
# Data rule:
#   start = max("2022-01-01", date when ALL tickers in group have data)
#   end   = "2025-10-23" (inclusive)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# If you don't have these:
# pip install yfinance cvxpy osqp
import yfinance as yf
import cvxpy as cp


# ----------------------------
# 1) Groups (your input)
# ----------------------------
groups = {
    "Eur":     ["FLIA", "BNDX", "ISHG", "BWX"],
    "War":     ["ITA", "IDEF", "PPA", "XAR"],
    "CybSeg":  ["CIBR", "HACK", "BUG", "IHAK", "WCBR"],
    "CloudCom":["IDGT", "SKYY", "CLOU", "XT"]
}

START_FLOOR = pd.Timestamp("2022-01-01")
END_DATE_INCLUSIVE = pd.Timestamp("2025-10-23")
END_DATE_EXCLUSIVE = END_DATE_INCLUSIVE + pd.Timedelta(days=1)  # yfinance end is exclusive

TRADING_DAYS = 252
N_FRONTIER_POINTS = 85  # increase if you want smoother
ADD_JITTER = True       # helps numerics (Sigma PSD)


# ----------------------------
# 2) Helpers: download + build data_by_group
# ----------------------------
def download_adj_close(tickers, start, end_exclusive):
    """
    Download Adj Close from yfinance for tickers in [start, end_exclusive).
    Returns a DataFrame with columns=tickers.
    """
    df = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end_exclusive.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"yfinance returned empty data for {tickers}")

    # If multiple tickers, yfinance returns multiindex columns; pick Adj Close
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" in df.columns.get_level_values(0)):
            px = df["Adj Close"].copy()
        else:
            # fallback if Adj Close missing for some reason
            px = df.xs("Close", axis=1, level=0)
    else:
        # single ticker case
        px = df.rename(columns={"Adj Close": tickers[0]}) if "Adj Close" in df.columns else df[["Close"]].rename(columns={"Close": tickers[0]})

    # Ensure columns order
    px = px.loc[:, [c for c in tickers if c in px.columns]]

    return px


def build_data_by_group(groups_dict):
    """
    For each group:
      - download prices from START_FLOOR to END_DATE_INCLUSIVE
      - compute common start date = latest first-valid date across tickers
      - final start date = max(START_FLOOR, common_start_date)
      - slice and drop any remaining NA rows (aligned panel)
    Returns:
      data_by_group: dict[str, pd.DataFrame] of Adj Close
      meta_by_group: dict[str, dict] with start_date, n_obs, tickers_used
    """
    data_by_group = {}
    meta_by_group = {}

    for g, tickers in groups_dict.items():
        print(f"\n--- Downloading group '{g}' ({len(tickers)} tickers) ---")

        px = download_adj_close(tickers, START_FLOOR, END_DATE_EXCLUSIVE)

        # Identify tickers with any valid data
        valid_cols = [c for c in px.columns if px[c].notna().any()]
        missing = [t for t in tickers if t not in valid_cols]
        if missing:
            print(f"  WARNING: No usable data for: {missing}")

        px = px[valid_cols].copy()

        if px.shape[1] < 2:
            raise ValueError(f"Group '{g}' has <2 tickers with data. Can't build covariance/frontier.")

        # Common start: latest first-valid date across tickers
        first_valids = {c: px[c].first_valid_index() for c in px.columns}
        common_start = max([d for d in first_valids.values() if d is not None])

        start_date = max(START_FLOOR, common_start)

        # Slice to your end date (inclusive), and align by dropping any row with NA
        px2 = px.loc[start_date:END_DATE_INCLUSIVE].dropna(axis=0, how="any")

        if px2.empty or len(px2) < 30:
            raise ValueError(
                f"Group '{g}' has too few aligned observations after slicing "
                f"({len(px2)} rows). Check tickers/data availability."
            )

        data_by_group[g] = px2
        meta_by_group[g] = {
            "start_date": start_date,
            "end_date": END_DATE_INCLUSIVE,
            "n_obs": len(px2),
            "tickers_used": list(px2.columns),
        }

        print(f"  Start used: {start_date.date()} | End used: {END_DATE_INCLUSIVE.date()} | Obs: {len(px2)}")
        print(f"  Tickers used: {list(px2.columns)}")

    return data_by_group, meta_by_group


# ----------------------------
# 3) Returns + (mu, Sigma)
# ----------------------------
def prices_to_returns(px: pd.DataFrame) -> pd.DataFrame:
    # simple returns
    rets = px.pct_change().dropna()
    return rets


def estimate_mu_sigma(rets: pd.DataFrame, annualize=True):
    """
    mu: expected returns vector (mean)
    Sigma: covariance matrix
    """
    mu = rets.mean()
    Sigma = rets.cov()

    if annualize:
        mu = mu * TRADING_DAYS
        Sigma = Sigma * TRADING_DAYS

    return mu.values, Sigma.values, list(rets.columns)


# ----------------------------
# 4) Efficient frontier solver (classic target return sweep)
# ----------------------------
def solve_gmv(mu, Sigma):
    """
    Global minimum variance portfolio (long-only, fully invested).
    """
    n = len(mu)
    w = cp.Variable(n)

    # Jitter helps PSD issues
    Sigma_use = Sigma.copy()
    if ADD_JITTER:
        Sigma_use = Sigma_use + 1e-10 * np.eye(n)

    objective = cp.Minimize(cp.quad_form(w, Sigma_use))
    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise RuntimeError("GMV optimization failed.")

    w_gmv = np.array(w.value).reshape(-1)
    return w_gmv


def solve_frontier(mu, Sigma, n_points=60):
    """
    Build classic efficient frontier:
      minimize variance s.t. mu'w >= R, sum w = 1, w >= 0
    Returns a DataFrame with columns: ret, vol, weights (list)
    """
    n = len(mu)

    # Compute GMV first to get a sensible minimum target return
    w_gmv = solve_gmv(mu, Sigma)
    ret_gmv = float(mu @ w_gmv)

    # Max return under long-only, fully invested (no caps) is simply max(mu)
    ret_max = float(np.max(mu))

    # Grid of target returns (avoid exactly ret_max for numerical stability)
    targets = np.linspace(ret_gmv, ret_max * 0.999, n_points)

    Sigma_use = Sigma.copy()
    if ADD_JITTER:
        Sigma_use = Sigma_use + 1e-10 * np.eye(n)

    out = []
    for R in targets:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, Sigma_use))
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            mu @ w >= R
        ]
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)

        if w.value is None:
            # skip infeasible/failed points
            continue

        w_opt = np.array(w.value).reshape(-1)
        pret = float(mu @ w_opt)
        pvol = float(np.sqrt(w_opt @ Sigma @ w_opt))

        out.append({"ret": pret, "vol": pvol, "weights": w_opt})

    df = pd.DataFrame(out)
    return df, w_gmv


# ----------------------------
# 5) Run everything per group + plot
# ----------------------------
def main():
    # Build prices per group
    data_by_group, meta_by_group = build_data_by_group(groups)

    # Store outputs
    returns_by_group = {}
    frontier_by_group = {}
    gmv_by_group = {}

    for g, px in data_by_group.items():
        rets = prices_to_returns(px)
        returns_by_group[g] = rets

        mu, Sigma, tickers_used = estimate_mu_sigma(rets, annualize=True)

        frontier_df, w_gmv = solve_frontier(mu, Sigma, n_points=N_FRONTIER_POINTS)

        # Save
        frontier_by_group[g] = (frontier_df, tickers_used, mu, Sigma)
        gmv_by_group[g] = (w_gmv, tickers_used)

        # Plot
        plt.figure()
        plt.plot(frontier_df["vol"], frontier_df["ret"], marker="o", linestyle="-")
        # Mark GMV explicitly
        ret_gmv = float(mu @ w_gmv)
        vol_gmv = float(np.sqrt(w_gmv @ Sigma @ w_gmv))
        plt.scatter([vol_gmv], [ret_gmv], s=120, marker="X")
        plt.title(f"Efficient Frontier (Long-Only) — {g}")
        plt.xlabel("Volatility (annualized)")
        plt.ylabel("Expected Return (annualized)")
        plt.grid(True)
        plt.show()

        # Print GMV weights
        w_series = pd.Series(w_gmv, index=tickers_used).sort_values(ascending=False)
        print(f"\nGMV weights for {g}:")
        print(w_series.round(4))

    return data_by_group, returns_by_group, frontier_by_group, gmv_by_group, meta_by_group


# Execute
if __name__ == "__main__":
    data_by_group, returns_by_group, frontier_by_group, gmv_by_group, meta_by_group = main()


######## Simple local plot ######3

import plotly.express as px

for g, pack in frontier_by_group.items():
    df = pack[0]          # <-- frontier_df
    df_plot = df[["vol", "ret"]].copy().sort_values("vol")

    fig = px.line(df_plot, x="vol", y="ret", markers=True,
                  title=f"Efficient Frontier — {g}")
    fig.update_layout(xaxis_title="Volatility (annualized)",
                      yaxis_title="Expected Return (annualized)")
    fig.write_html(f"frontier_{g}.html", auto_open=True)

print("Done. Open frontier_Eur.html (etc.) in your browser.")


###############3 Using Streamlit ################


# frontier_by_group[g] = (frontier_df, tickers_used, mu, Sigma)
for g, pack in frontier_by_group.items():
    df = pack[0].copy()
    tickers = pack[1]

    W = np.vstack(df["weights"].to_numpy())  # shape: (n_points, n_assets)
    dfw = pd.DataFrame(W, columns=tickers, index=df.index)

    out = pd.concat([df[["vol", "ret"]].reset_index(drop=True),
                     dfw.reset_index(drop=True)], axis=1)

    out.to_parquet(f"frontier_{g}.parquet", index=False)

print("Saved: frontier_<group>.parquet files")

import glob
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Efficient Frontier", layout="wide")
st.title("Efficient Frontier — Click a point to see weights")

# --- discover available groups from files
files = sorted(glob.glob("frontier_*.parquet"))
if not files:
    st.error("No frontier_*.parquet files found. Run the notebook export step first.")
    st.stop()

groups = [os.path.basename(f).replace("frontier_", "").replace(".parquet", "") for f in files]
g = st.sidebar.selectbox("Group", groups)

df = pd.read_parquet(f"frontier_{g}.parquet")
df = df.sort_values("vol").reset_index(drop=True)

ticker_cols = [c for c in df.columns if c not in ["vol", "ret"]]

# --- build plotly figure (line + markers)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df["vol"],
        y=df["ret"],
        mode="lines+markers",
        # put a compact hover; full weights go in table after click
        hovertemplate="Vol: %{x:.4f}<br>Ret: %{y:.4f}<extra></extra>",
    )
)
fig.update_layout(
    title=f"Efficient Frontier — {g}",
    xaxis_title="Volatility (annualized)",
    yaxis_title="Expected Return (annualized)",
)

left, right = st.columns([2, 1])

with left:
    event = st.plotly_chart(
        fig,
        key=f"frontier_{g}",
        on_select="rerun",
        selection_mode="points",
        height=550,
    )

# --- determine selected point
selected_idx = None
if event and event.selection and event.selection.get("point_indices"):
    selected_idx = event.selection["point_indices"][0]  # first selected point

# default selection: min-vol point
if selected_idx is None:
    selected_idx = int(df["vol"].idxmin())

row = df.iloc[selected_idx]

weights = (
    pd.DataFrame({"ticker": ticker_cols, "weight": row[ticker_cols].values})
    .sort_values("weight", ascending=False)
    .reset_index(drop=True)
)

with right:
    st.subheader("Selected point")
    st.write({"vol": float(row["vol"]), "ret": float(row["ret"]), "point_index": int(selected_idx)})

    st.subheader("Weights")
    st.dataframe(weights, use_container_width=True, height=420)

    st.caption("Tip: click a marker on the chart to update this table.")

wkdir = os.getcwd()

streamlit run app.py
