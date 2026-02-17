#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 00:40:09 2026

@author: hernanvillanueva

efficient_frontier_backend.py

Backend utilities for the Streamlit Efficient Frontier app.

Design goals (aligned with FronteraEficiente_Yields.py):
  - Long-only, fully-invested portfolios.
  - Efficient frontier solved via *target return sweep*:
        min  w'Σw   s.t.  1'w=1, w>=0, μ'w >= R
  - Optional global per-asset cap: w_i <= cap (cap in [0.30, 1.00])
  - Annualized μ and Σ from daily returns (252 trading days).
  - Optional EUR yield-proxy μ (SEC yield - expense ratio), while Σ stays historical.
  - Slope≈1 anchor computed via finite differences on discrete frontier points,
    after sorting by vol and de-duplicating equal vols.

This module intentionally has **no Streamlit imports**. Use Streamlit caching
in the app layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import cvxpy as cp
import yfinance as yf


# -----------------------------------------------------------------------------
# Configuration (kept consistent with your notebook/script defaults)
# -----------------------------------------------------------------------------

TRADING_DAYS: int = 252

# Data window rule:
START_FLOOR = pd.Timestamp("2022-01-01")
END_DATE_INCLUSIVE = pd.Timestamp("2025-10-23")

# Numerical stability
ADD_JITTER: bool = True
JITTER_EPS: float = 1e-10


# Groups (same tickers as your generator)
GROUPS: Dict[str, List[str]] = {
    "Eur": ["FLIA", "BNDX", "ISHG", "BWX"],
    "War": ["ITA", "IDEF", "PPA", "XAR"],
    "CybSeg": ["CIBR", "HACK", "BUG", "IHAK", "WCBR"],
    "CloudCom": ["IDGT", "SKYY", "CLOU", "XT"],
}


# Expected return overrides (Yield Proxy) — annual decimals
YIELD_PROXY: Dict[str, Dict[str, Dict[str, float]]] = {
    "Eur": {
        "FLIA": {"sec_yield": 0.0257, "expense_ratio": 0.0025},
        "BNDX": {"sec_yield": 0.0314, "expense_ratio": 0.0007},
        "ISHG": {"sec_yield": 0.0199, "expense_ratio": 0.0035},
        "BWX": {"sec_yield": 0.0275, "expense_ratio": 0.0035},
    }
}


# Reference tickers (set to your "same as before" defaults).
# If your "before" set differs, change it here once and the app will follow.
REFERENCE_TICKERS: Tuple[str, str, str] = ("SPY", "QQQ", "DXY")


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontierResult:
    """Container for one computed frontier."""

    group: str
    tickers: List[str]
    mu_method: str  # "historical" or "yield_proxy"
    cap: float
    n_points: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    mu: np.ndarray
    Sigma: np.ndarray
    frontier: pd.DataFrame  # includes Frontier + anchors, with scalar weight columns
    slope_anchor: Optional[pd.Series]  # row-like series with vol/ret/slope + weights


# -----------------------------------------------------------------------------
# Price download + alignment
# -----------------------------------------------------------------------------


def download_adj_close(
    tickers: Iterable[str],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Download Adj Close for tickers in [start, end_exclusive) using yfinance."""
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers must be non-empty")

    df = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end_exclusive.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"yfinance returned empty data for tickers={tickers}")

    # MultiIndex columns when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            px = df["Adj Close"].copy()
        else:
            px = df.xs("Close", axis=1, level=0).copy()
    else:
        # Single ticker
        t0 = tickers[0]
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]].rename(columns={"Adj Close": t0})
        else:
            px = df[["Close"]].rename(columns={"Close": t0})

    # Keep requested order, drop missing columns
    px = px.loc[:, [t for t in tickers if t in px.columns]]
    return px


def align_prices_common_start(
    px_raw: pd.DataFrame,
    tickers: List[str],
    start_floor: pd.Timestamp = START_FLOOR,
    end_inclusive: pd.Timestamp = END_DATE_INCLUSIVE,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Apply your rule: start = max(start_floor, common first-valid), end=end_inclusive."""
    if px_raw.empty:
        raise ValueError("px_raw is empty")

    # Drop tickers with no data
    valid_cols = [c for c in tickers if c in px_raw.columns and px_raw[c].notna().any()]
    if len(valid_cols) < 2:
        raise ValueError("Need at least 2 tickers with usable data after download/alignment")

    px_raw = px_raw[valid_cols].copy()
    first_valids = {c: px_raw[c].first_valid_index() for c in valid_cols}
    common_start = max([d for d in first_valids.values() if d is not None])
    start_date = max(start_floor, common_start)

    px = px_raw.loc[start_date:end_inclusive].dropna(axis=0, how="any")
    if len(px) < 30:
        raise ValueError(f"Too few aligned rows after slicing: {len(px)}")
    return px, start_date


def load_group_prices_from_parquet(
    group: str,
    parquet_dir: str = ".",
) -> pd.DataFrame:
    """Load aligned group prices from prices_<group>.parquet (as exported by your generator)."""
    path = f"{parquet_dir.rstrip('/')}/prices_{group}.parquet"
    px = pd.read_parquet(path)
    if "date" in px.columns:
        px["date"] = pd.to_datetime(px["date"])
        px = px.set_index("date")
    px = px.sort_index()
    return px


# -----------------------------------------------------------------------------
# Estimation (μ, Σ)
# -----------------------------------------------------------------------------


def estimate_mu_sigma(
    prices: pd.DataFrame,
    tickers: List[str],
    group: str,
    mu_method: str = "historical",
    trading_days: int = TRADING_DAYS,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimate annualized mu and Sigma from daily returns (or yield proxy for Eur)."""

    prices = prices.loc[:, tickers].copy()
    rets = prices.pct_change().dropna()

    Sigma = (rets.cov() * trading_days).to_numpy(dtype=float)

    if mu_method == "historical":
        mu = (rets.mean() * trading_days).to_numpy(dtype=float)
    elif mu_method == "yield_proxy":
        if group not in YIELD_PROXY:
            raise ValueError(f"Yield proxy requested, but group '{group}' has no YIELD_PROXY inputs")
        yi = YIELD_PROXY[group]
        yields_tbl = pd.DataFrame(yi).T
        yields_tbl.columns = [c.strip().lower() for c in yields_tbl.columns]
        if "sec_yield" not in yields_tbl.columns or "expense_ratio" not in yields_tbl.columns:
            raise KeyError(
                f"[{group}] Yield proxy table must include sec_yield and expense_ratio. Got {list(yields_tbl.columns)}"
            )
        yields_tbl["net_yield"] = yields_tbl["sec_yield"] - yields_tbl["expense_ratio"]
        missing = [t for t in tickers if t not in yields_tbl.index]
        if missing:
            raise ValueError(f"[{group}] Missing yield inputs for: {missing}")
        mu = yields_tbl.loc[tickers, "net_yield"].to_numpy(dtype=float)
    else:
        raise ValueError("mu_method must be 'historical' or 'yield_proxy'")

    return mu, Sigma, rets


# -----------------------------------------------------------------------------
# Optimizers (cvxpy)
# -----------------------------------------------------------------------------


def _solve_problem(prob: cp.Problem) -> None:
    """Solve with OSQP then SCS fallback."""
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)


def solve_gmv(mu: np.ndarray, Sigma: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """GMV: min w'Σw s.t. 1'w=1, w>=0, (optional) w<=cap."""
    n = len(mu)
    w = cp.Variable(n)

    Sigma_use = Sigma.copy()
    if ADD_JITTER:
        Sigma_use = Sigma_use + JITTER_EPS * np.eye(n)

    cons = [cp.sum(w) == 1, w >= 0]
    if cap < 1.0 - 1e-12:
        cons.append(w <= cap)

    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma_use)), cons)
    _solve_problem(prob)

    if w.value is None:
        raise RuntimeError("GMV optimization failed.")
    return np.array(w.value, dtype=float).reshape(-1)


def solve_max_return(mu: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """Max-return under long-only + global cap.

    For cap=1.0, this matches your file: one-hot on argmax(mu).
    For cap<1.0, this is the exact solution for: max mu'w s.t. sum(w)=1, 0<=w<=cap.
    """
    n = len(mu)
    mu = np.array(mu, dtype=float).reshape(-1)

    if cap >= 1.0 - 1e-12:
        w = np.zeros(n, dtype=float)
        w[int(np.argmax(mu))] = 1.0
        return w

    # Greedy fill is optimal for this constraint set.
    order = np.argsort(-mu)  # descending mu
    w = np.zeros(n, dtype=float)
    remaining = 1.0
    for idx in order:
        if remaining <= 0:
            break
        alloc = min(cap, remaining)
        w[idx] = alloc
        remaining -= alloc

    # Should be feasible when n*cap >= 1
    if abs(w.sum() - 1.0) > 1e-8:
        raise RuntimeError(
            f"Max-return allocation infeasible for cap={cap:.4f} with n={n} (sum={w.sum():.6f})."
        )
    return w


def solve_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_points: int,
    cap: float = 1.0,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Efficient frontier via target return sweep (same as your file), with optional cap."""
    mu = np.array(mu, dtype=float).reshape(-1)
    Sigma = np.array(Sigma, dtype=float)
    n = len(mu)

    w_gmv = solve_gmv(mu, Sigma, cap=cap)
    ret_gmv = float(mu @ w_gmv)

    w_max = solve_max_return(mu, cap=cap)
    ret_max = float(mu @ w_max)

    hi = ret_max * 0.999
    if hi < ret_gmv:
        # Numerical edge case; make a tiny upward range
        hi = ret_gmv * (1.0 + 1e-6)
    targets = np.linspace(ret_gmv, hi, int(n_points))

    Sigma_use = Sigma.copy()
    if ADD_JITTER:
        Sigma_use = Sigma_use + JITTER_EPS * np.eye(n)

    rows = []
    for R in targets:
        w = cp.Variable(n)
        cons = [cp.sum(w) == 1, w >= 0, mu @ w >= float(R)]
        if cap < 1.0 - 1e-12:
            cons.append(w <= cap)

        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma_use)), cons)
        _solve_problem(prob)
        if w.value is None:
            continue

        w_opt = np.array(w.value, dtype=float).reshape(-1)
        pret = float(mu @ w_opt)
        pvol = float(np.sqrt(w_opt @ Sigma @ w_opt))
        rows.append({"point_type": "Frontier", "ret": pret, "vol": pvol, "weights": w_opt})

    frontier_df = pd.DataFrame(rows)
    return frontier_df, w_gmv, w_max


# -----------------------------------------------------------------------------
# Slope≈1 anchor
# -----------------------------------------------------------------------------


def compute_slope_anchor(
    frontier_df: pd.DataFrame,
    tickers: List[str],
    target_slope: float = 1.0,
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """Compute slope series and the 'closest to slope=1' anchor.

    Returns:
      anchor_row: pd.Series with columns: vol, ret, slope, and ticker weights
      slope_df:   frontier points with slope columns (deduped-by-vol)

    Tie-break: if multiple are equally close to target_slope, pick the lowest vol.
    """
    if frontier_df is None or frontier_df.empty:
        return None, pd.DataFrame()

    d = frontier_df.copy()
    # Ensure types
    d["vol"] = pd.to_numeric(d["vol"], errors="coerce")
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce")
    d = d.dropna(subset=["vol", "ret"]).copy()

    # Ensure weight columns exist
    missing_w = [t for t in tickers if t not in d.columns]
    if missing_w:
        # If only vector weights exist, expand
        if "weights" not in d.columns:
            raise KeyError(f"Frontier df missing ticker weight columns and 'weights' vector: {missing_w}")
        W = np.vstack(d["weights"].to_numpy())
        for j, t in enumerate(tickers):
            d[t] = W[:, j]

    # Clean + sort
    d = d.sort_values(["vol", "ret"], ascending=[True, False])
    d = d.drop_duplicates(subset=["vol"], keep="first")
    d = d.sort_values("vol", ascending=True).reset_index(drop=True)

    n = len(d)
    d["slope_raw"] = np.nan
    d["slope"] = np.nan
    if n >= 2:
        vol = d["vol"].to_numpy(float)
        ret = d["ret"].to_numpy(float)
        slope = np.full(n, np.nan, float)

        dv0 = vol[1] - vol[0]
        if abs(dv0) > 1e-12:
            slope[0] = (ret[1] - ret[0]) / dv0

        dvn = vol[-1] - vol[-2]
        if abs(dvn) > 1e-12:
            slope[-1] = (ret[-1] - ret[-2]) / dvn

        if n >= 3:
            dv = vol[2:] - vol[:-2]
            dr = ret[2:] - ret[:-2]
            ok = np.abs(dv) > 1e-12
            slope[1:-1][ok] = dr[ok] / dv[ok]

        d["slope_raw"] = slope
        d["slope"] = d["slope_raw"]

    if not d["slope"].notna().any():
        return None, d

    closeness = (d["slope"] - float(target_slope)).abs()
    min_close = float(closeness.min())
    cand = d.loc[closeness == min_close].copy()
    cand = cand.sort_values("vol", ascending=True)
    anchor = cand.iloc[0]
    return anchor, d


# -----------------------------------------------------------------------------
# Output shaping for Streamlit
# -----------------------------------------------------------------------------


def pack_frontier_output(
    frontier_only: pd.DataFrame,
    tickers: List[str],
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_gmv: np.ndarray,
    w_max: np.ndarray,
    slope_anchor: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build a single dataframe with scalar ticker columns (as your Streamlit expects)."""
    # Frontier points
    df = frontier_only.copy()
    if df.empty:
        raise ValueError("frontier_only is empty")

    # Expand weights vector into columns if needed
    if "weights" in df.columns and not all(t in df.columns for t in tickers):
        W = np.vstack(df["weights"].to_numpy())
        wcols = pd.DataFrame(W, columns=tickers)
        df = pd.concat([df.drop(columns=["weights"]), wcols], axis=1)

    # Anchors
    ret_gmv = float(mu @ w_gmv)
    vol_gmv = float(np.sqrt(w_gmv @ Sigma @ w_gmv))
    ret_max = float(mu @ w_max)
    vol_max = float(np.sqrt(w_max @ Sigma @ w_max))

    anchors = [
        {"point_type": "GMV", "ret": ret_gmv, "vol": vol_gmv, **{t: float(v) for t, v in zip(tickers, w_gmv)}},
        {"point_type": "MaxReturn", "ret": ret_max, "vol": vol_max, **{t: float(v) for t, v in zip(tickers, w_max)}},
    ]
    if slope_anchor is not None and isinstance(slope_anchor, pd.Series):
        anchors.append(
            {
                "point_type": "Slope1",
                "ret": float(slope_anchor["ret"]),
                "vol": float(slope_anchor["vol"]),
                "slope": float(slope_anchor.get("slope", np.nan)),
                **{t: float(slope_anchor[t]) for t in tickers},
            }
        )

    out = pd.concat([df, pd.DataFrame(anchors)], ignore_index=True)
    # Standard column order
    base_cols = ["point_type", "vol", "ret"]
    extra = [c for c in out.columns if c not in base_cols + tickers]
    out = out[base_cols + extra + tickers]
    return out


def compute_frontier_for_group(
    group: str,
    cap: float,
    n_points: int,
    mu_method: str = "historical",
    prices: Optional[pd.DataFrame] = None,
    parquet_dir: str = ".",
    end_date_inclusive: pd.Timestamp = END_DATE_INCLUSIVE,
) -> FrontierResult:
    """High-level helper used by the Streamlit app."""
    if group not in GROUPS:
        raise KeyError(f"Unknown group '{group}'. Options: {list(GROUPS)}")
    if not (0 < cap <= 1.0):
        raise ValueError("cap must be in (0, 1]")

    tickers = list(GROUPS[group])

    if prices is None:
        prices = load_group_prices_from_parquet(group, parquet_dir=parquet_dir)

    # Respect the fixed estimation window end
    prices = prices.loc[:end_date_inclusive].copy()
    tickers_used = [t for t in tickers if t in prices.columns]
    if len(tickers_used) < 2:
        raise ValueError(f"Group '{group}' has <2 tickers in provided prices")

    # Feasibility: n*cap >= 1
    if len(tickers_used) * cap < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible constraints: n_assets={len(tickers_used)} cap={cap:.3f} => n*cap={len(tickers_used)*cap:.3f} < 1"
        )

    mu, Sigma, _rets = estimate_mu_sigma(
        prices=prices,
        tickers=tickers_used,
        group=group,
        mu_method=mu_method,
        trading_days=TRADING_DAYS,
    )

    frontier_only, w_gmv, w_max = solve_frontier(mu, Sigma, n_points=n_points, cap=cap)
    # compute slope anchor on the expanded-weight version
    # (anchor weights are required for the app)
    frontier_only_expanded = pack_frontier_output(
        frontier_only=frontier_only,
        tickers=tickers_used,
        mu=mu,
        Sigma=Sigma,
        w_gmv=w_gmv,
        w_max=w_max,
        slope_anchor=None,
    )
    f_only = frontier_only_expanded[frontier_only_expanded["point_type"].eq("Frontier")].copy()
    slope_anchor, _slope_df = compute_slope_anchor(f_only, tickers_used, target_slope=1.0)

    frontier_full = pack_frontier_output(
        frontier_only=frontier_only,
        tickers=tickers_used,
        mu=mu,
        Sigma=Sigma,
        w_gmv=w_gmv,
        w_max=w_max,
        slope_anchor=slope_anchor,
    )

    return FrontierResult(
        group=group,
        tickers=tickers_used,
        mu_method=mu_method,
        cap=float(cap),
        n_points=int(n_points),
        start_date=pd.Timestamp(prices.index.min()),
        end_date=pd.Timestamp(prices.index.max()),
        mu=mu,
        Sigma=Sigma,
        frontier=frontier_full,
        slope_anchor=slope_anchor,
    )


# -----------------------------------------------------------------------------
# Performance (buy-and-hold base 100)
# -----------------------------------------------------------------------------


def base100_buy_and_hold(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Base-100 for buy-and-hold (no rebalancing), using initial value weights."""
    w = weights.copy().astype(float)
    w = w[w.index.intersection(prices.columns)]
    w = w / w.sum()
    px = prices.loc[:, w.index]
    rel = px / px.iloc[0]
    return 100.0 * rel.mul(w, axis=1).sum(axis=1)


def base100_single(prices: pd.Series) -> pd.Series:
    """Base-100 for a single asset."""
    s = prices.dropna().copy()
    return 100.0 * (s / s.iloc[0])


def fetch_prices_for_live_window(
    tickers: List[str],
    start_inclusive: pd.Timestamp,
    end_inclusive: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Fetch Adj Close for live performance; returns a dataframe indexed by date."""
    if end_inclusive is None:
        end_excl = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    else:
        end_excl = pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)

    px = download_adj_close(tickers, start_inclusive, end_excl)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    # Ensure same trading calendar (intersection) by dropping any rows with missing values
    px = px.dropna(axis=0, how="any")
    return px
