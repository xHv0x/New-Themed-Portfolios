#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""app_v5.py

Efficient Frontier — Generator (Streamlit)

v5 goals (aligned with FronteraEficiente_Yields.py):
- Compute efficient frontier on the fly (target-return sweep).
- Long-only, fully-invested portfolios.
- Optional global per-asset cap (slider 0.30 → 1.00; 1.00 means unconstrained).
- Highlight GMV, MaxReturn, and slope≈1 (Δret/Δvol closest to 1).
- Use Adj Close and buy-and-hold (no rebalancing) for Base-100 performance.
- Frontier estimation window is fixed to END_DATE_INCLUSIVE = 2025-10-23 (inclusive).
- "Live performance" (out-of-sample since cutoff) is shown only on button click.
- Selection resets to blank when frontier is recomputed.

Requires:
- efficient_frontier_backend.py in the same folder.
- prices_<group>.parquet files (optional; app can fallback to yfinance).

"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import efficient_frontier_backend as efb


# -----------------------------------------------------------------------------
# Frontier density policy
# -----------------------------------------------------------------------------

TARGET_FRONTIER_POINTS = 60
MIN_FRONTIER_POINTS = 40


# -----------------------------------------------------------------------------
# Page config + compact styling
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Efficient Frontier (v5)", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.0rem; padding-bottom: 1.0rem;}
      h1, h2, h3 {margin-top: 0.2rem; margin-bottom: 0.4rem;}
      div[data-testid="stMetricValue"] {font-size: 2.0rem;}
      .small-note {color: rgba(49, 51, 63, 0.7); font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Efficient Frontier — Generator (v5)")


# -----------------------------------------------------------------------------
# Reference tickers ("same as before")
# NOTE: yfinance symbol for DXY is commonly "DX-Y.NYB".
# We keep the display label as "DXY" while fetching via the yfinance symbol.
# If your prior setup used a different symbol, change it here once.
# -----------------------------------------------------------------------------

REFERENCE_DISPLAY_TO_YF: Dict[str, str] = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "DXY": "DX-Y.NYB",  # display label "DXY" → yfinance "DX-Y.NYB"
}
REFERENCE_DISPLAY_ORDER: Tuple[str, str, str] = efb.REFERENCE_TICKERS


# -----------------------------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_group_prices_cached(group: str, parquet_dir: str = ".") -> pd.DataFrame:
    """Load prices_<group>.parquet if present; fallback to yfinance + alignment."""
    parquet_path = f"{parquet_dir.rstrip('/')}/prices_{group}.parquet"

    if os.path.exists(parquet_path):
        px = pd.read_parquet(parquet_path)
        if "date" in px.columns:
            px["date"] = pd.to_datetime(px["date"])
            px = px.set_index("date")
        px = px.sort_index()
        return px

    # Fallback: download and align using the same rules as your generator
    tickers = efb.GROUPS[group]
    end_excl = efb.END_DATE_INCLUSIVE + pd.Timedelta(days=1)
    px_raw = efb.download_adj_close(tickers, efb.START_FLOOR, end_excl)
    px, _start_date = efb.align_prices_common_start(
        px_raw=px_raw,
        tickers=tickers,
        start_floor=efb.START_FLOOR,
        end_inclusive=efb.END_DATE_INCLUSIVE,
    )
    return px


@st.cache_data(show_spinner=False)
def _compute_frontier_cached(
    group: str,
    cap: float,
    n_points: int,
    mu_method: str,
    parquet_dir: str = ".",
) -> efb.FrontierResult:
    px = _load_group_prices_cached(group, parquet_dir=parquet_dir)
    return efb.compute_frontier_for_group(
        group=group,
        cap=float(cap),
        n_points=int(n_points),
        mu_method=mu_method,
        prices=px,
        parquet_dir=parquet_dir,
        end_date_inclusive=efb.END_DATE_INCLUSIVE,
    )


@st.cache_data(show_spinner=False)
def _fetch_reference_prices_in_sample(
    display_tickers: Tuple[str, ...],
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch Adj Close references for the in-sample window."""
    yf_tickers = [REFERENCE_DISPLAY_TO_YF.get(t, t) for t in display_tickers]
    end_excl = end_inclusive + pd.Timedelta(days=1)
    px = efb.download_adj_close(yf_tickers, start, end_excl)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    # Rename columns back to display tickers if yfinance symbols differ
    ren = {REFERENCE_DISPLAY_TO_YF.get(t, t): t for t in display_tickers}
    px = px.rename(columns=ren)
    return px


@st.cache_data(show_spinner=False)
def _fetch_live_prices(
    tickers: Tuple[str, ...],
    start_inclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch Adj Close for live performance, aligned by dropping rows with any NA."""
    px = efb.fetch_prices_for_live_window(list(tickers), start_inclusive=start_inclusive)
    return px


# -----------------------------------------------------------------------------
# UI — Sidebar
# -----------------------------------------------------------------------------

st.sidebar.markdown("### Settings")

GROUPS = list(efb.GROUPS.keys())

group = st.sidebar.selectbox("Group", GROUPS, index=0)

# EUR expected return source (only meaningful for Eur)
if group == "Eur":
    mu_method = st.sidebar.radio(
        "EUR expected return source",
        options=["historical", "yield_proxy"],
        index=0,
        format_func=lambda x: "Historical returns" if x == "historical" else "SEC yield proxy",
    )
else:
    mu_method = "historical"
    st.sidebar.caption("EUR yield proxy is only available for the Eur group.")

cap = st.sidebar.slider(
    "Max weight per asset (cap)",
    min_value=0.30,
    max_value=1.00,
    value=1.00,
    step=0.01,
    help="Global cap applied to every asset weight. 1.00 means unconstrained.",
)

st.sidebar.divider()

recompute = st.sidebar.button("Recompute frontier", type="primary", use_container_width=True)

st.sidebar.markdown(
    f"<div class='small-note'>Estimation window ends on <b>{efb.END_DATE_INCLUSIVE.date()}</b> (inclusive).</div>",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------


def _current_frontier_key() -> Tuple:
    return (group, mu_method, round(float(cap), 4))


if "frontier_key" not in st.session_state:
    st.session_state.frontier_key = None

if "frontier_result" not in st.session_state:
    st.session_state.frontier_result = None

if "frontier_front_points" not in st.session_state:
    st.session_state.frontier_front_points = None

if "frontier_requested_points" not in st.session_state:
    st.session_state.frontier_requested_points = None

if "selected_point_idx" not in st.session_state:
    st.session_state.selected_point_idx = None

if "sel_point" not in st.session_state:
    st.session_state.sel_point = "—"

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Frontier"  # or "Base100"

if "show_live" not in st.session_state:
    st.session_state.show_live = False


def _reset_selection_after_recompute():
    st.session_state.selected_point_idx = None
    st.session_state.show_live = False
    # Keep selection widgets in sync
    st.session_state.sel_point = "—"


def _count_frontier_points(fr: efb.FrontierResult) -> int:
    d = fr.frontier
    if d is None or d.empty or "point_type" not in d.columns:
        return 0
    return int(d["point_type"].eq("Frontier").sum())


def _compute_with_fallback(group: str, cap: float, mu_method: str) -> efb.FrontierResult:
    """Try to compute a dense frontier; fallback from 60 down to 40.

    Policy:
      - Prefer 60 points.
      - If solver fails to produce all points, try fewer targets.
      - Track how many Frontier points we actually got.
      - If < 40, app warns.
    """
    best_fr: Optional[efb.FrontierResult] = None
    best_count = -1
    best_n = None

    for n in range(TARGET_FRONTIER_POINTS, MIN_FRONTIER_POINTS - 1, -1):
        fr_try = _compute_frontier_cached(
            group=group,
            cap=float(cap),
            n_points=int(n),
            mu_method=mu_method,
            parquet_dir=".",
        )
        c = _count_frontier_points(fr_try)
        if c > best_count:
            best_fr, best_count, best_n = fr_try, c, n
        # "Perfect" run: all requested targets produced a frontier point
        if c == n:
            st.session_state.frontier_front_points = c
            st.session_state.frontier_requested_points = n
            return fr_try

    # If we get here, keep the best attempt
    if best_fr is None:
        raise RuntimeError("Could not compute any frontier points.")

    st.session_state.frontier_front_points = best_count
    st.session_state.frontier_requested_points = best_n
    return best_fr


# -----------------------------------------------------------------------------
# Recompute logic (button-centered)
# -----------------------------------------------------------------------------

if recompute or (st.session_state.frontier_key is None):
    with st.spinner("Computing frontier…"):
        try:
            fr = _compute_with_fallback(group=group, cap=float(cap), mu_method=mu_method)
        except Exception as e:
            st.error(f"Could not compute frontier: {e}")
            st.stop()

    st.session_state.frontier_result = fr
    st.session_state.frontier_key = _current_frontier_key()
    _reset_selection_after_recompute()


# If user changed controls but hasn't recomputed, warn and keep showing last result
controls_key = _current_frontier_key()
using_key = st.session_state.frontier_key
fr: efb.FrontierResult = st.session_state.frontier_result

if fr is None:
    st.error("Frontier not computed yet. Click 'Recompute frontier'.")
    st.stop()

if using_key != controls_key:
    st.warning("Settings changed. Click **Recompute frontier** to update the frontier.")


# -----------------------------------------------------------------------------
# Helpers for rendering
# -----------------------------------------------------------------------------


def _ticker_cols(fr: efb.FrontierResult) -> List[str]:
    return list(fr.tickers)


def _weights_from_row(row: pd.Series, tickers: List[str]) -> pd.Series:
    w = pd.Series({t: float(row[t]) for t in tickers}, dtype=float)
    # cleanup + renormalize
    w[(w.abs() < 1e-12)] = 0.0
    w[w < 0] = 0.0
    s = float(w.sum())
    if s > 0:
        w = w / s
    return w


def _weights_table(w: pd.Series) -> pd.DataFrame:
    df = w.sort_values(ascending=False).reset_index()
    df.columns = ["ticker", "weight"]
    disp = df.copy()
    disp["weight"] = disp["weight"].map(lambda x: f"{x:.2%}")
    return df, disp


def _table_height_for(n_rows: int) -> int:
    h = 35 * (n_rows + 1) + 20
    return max(150, min(260, h))


def _point_stats(row: pd.Series) -> Tuple[float, float, float]:
    ret = float(row["ret"])
    vol = float(row["vol"])
    ratio = ret / vol if vol > 0 else np.nan
    return ret, vol, ratio


def _render_point(title: str, row: pd.Series, tickers: List[str]):
    st.markdown(f"### {title}")
    ret, vol, ratio = _point_stats(row)
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected return", f"{ret:.2%}")
    c2.metric("Volatility", f"{vol:.2%}")
    c3.metric("Ret/Vol", f"{ratio:.2f}" if np.isfinite(ratio) else "—")

    w = _weights_from_row(row, tickers)
    wdf, wdisp = _weights_table(w)
    st.dataframe(wdisp, use_container_width=True, height=_table_height_for(len(wdisp)), hide_index=True)
    st.caption(f"Weight sum: {float(w.sum()):.4f}")


def _make_frontier_figure(frontier_df: pd.DataFrame, title: str) -> go.Figure:
    d_all = frontier_df.copy()

    d_front = d_all[d_all["point_type"].eq("Frontier")].copy()
    d_front = d_front.sort_values("vol").reset_index(drop=True)

    def _first(ptype: str) -> Optional[pd.Series]:
        sub = d_all[d_all["point_type"].eq(ptype)]
        return sub.iloc[0] if not sub.empty else None

    row_gmv = _first("GMV")
    row_max = _first("MaxReturn")
    row_s1 = _first("Slope1")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d_front["vol"],
            y=d_front["ret"],
            mode="lines+markers",
            name="Frontier",
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
        )
    )

    if row_gmv is not None:
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

    if row_max is not None:
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

    if row_s1 is not None:
        s_txt = "Slope≈1"
        if "slope" in row_s1.index and pd.notna(row_s1["slope"]):
            s_txt = f"Slope≈{float(row_s1['slope']):.2f}"
        fig.add_trace(
            go.Scatter(
                x=[float(row_s1["vol"])],
                y=[float(row_s1["ret"])],
                mode="markers+text",
                name="Slope=1",
                marker_symbol="star",
                marker_size=12,
                text=[s_txt],
                textposition="top center",
                hovertemplate="Slope=1<br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Volatility (annualized)",
        yaxis_title="Expected Return (annualized)",
        legend_title="",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")

    return fig


# -----------------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------------

subtitle = ""
if group == "Eur":
    subtitle = " — SEC yield proxy" if mu_method == "yield_proxy" else " — historical"

front_count = st.session_state.frontier_front_points
req_count = st.session_state.frontier_requested_points
front_note = ""
if isinstance(front_count, int) and isinstance(req_count, int):
    if front_count < MIN_FRONTIER_POINTS:
        front_note = f"  |  ⚠️ computed **{front_count}** points (below {MIN_FRONTIER_POINTS})"
    elif front_count < TARGET_FRONTIER_POINTS:
        front_note = f"  |  computed **{front_count}** points"
    else:
        front_note = f"  |  computed **{front_count}** points"

st.caption(
    f"Group: **{group}**{subtitle}  |  cap: **{cap:.2f}**  |  cutoff: **{efb.END_DATE_INCLUSIVE.date()}**{front_note}"
)

if isinstance(front_count, int) and front_count < MIN_FRONTIER_POINTS:
    st.warning(
        f"Only **{front_count}** frontier points were produced (target is {TARGET_FRONTIER_POINTS}). "
        "The curve may look coarse; consider loosening constraints (higher cap) or switching group/method."
    )

left, right = st.columns([2.2, 1.0], gap="large")

# View toggle buttons (frontier vs base100)
with left:
    b1, b2 = st.columns([1, 1])
    if b1.button("Frontier view", use_container_width=True):
        st.session_state.view_mode = "Frontier"
    if b2.button("Base 100 view", use_container_width=True):
        st.session_state.view_mode = "Base100"


# Frontier plot + selection
frontier_df = fr.frontier.copy()
tickers = _ticker_cols(fr)

# ensure point types exist
if "point_type" not in frontier_df.columns:
    frontier_df["point_type"] = "Frontier"

# Primary frontier series for selection mapping
frontier_only = frontier_df[frontier_df["point_type"].eq("Frontier")].copy()
frontier_only = frontier_only.sort_values("vol").reset_index(drop=True)

fig = _make_frontier_figure(
    frontier_df=frontier_df,
    title=f"Efficient Frontier — {group}{subtitle}",
)

with left:
    if st.session_state.view_mode == "Frontier":
        st.plotly_chart(
            fig,
            key=f"frontier_v5_{group}_{mu_method}_{using_key}",
            height=520,
            use_container_width=True,
        )

        if frontier_only.empty:
            st.error("No frontier points were produced. Try a different cap or group.")
        else:
            opts = ["—"] + list(range(len(frontier_only)))

            def _fmt(opt):
                if opt == "—":
                    return "—"
                i = int(opt)
                r = frontier_only.iloc[i]
                return f"#{i:02d}  |  vol {float(r['vol']):.2%}  |  ret {float(r['ret']):.2%}"

            sel = st.selectbox(
                "Select a frontier point (0 = lowest vol)",
                options=opts,
                key="sel_point",
                format_func=_fmt,
                help="Frontier points are not clickable. Use this selector to choose a portfolio point.",
            )

            prev = st.session_state.selected_point_idx
            if sel == "—":
                if prev is not None:
                    st.session_state.selected_point_idx = None
                    st.session_state.show_live = False
            else:
                idx = int(sel)
                if prev != idx:
                    st.session_state.selected_point_idx = idx
                    st.session_state.show_live = False

            if st.session_state.selected_point_idx is None:
                st.info("Frontier updated — please select a portfolio point.")

    else:
        if st.session_state.selected_point_idx is None:
            st.info("Select a frontier point first to view Base 100.")
        else:
            sel_idx = int(st.session_state.selected_point_idx)
            sel_idx = max(0, min(sel_idx, len(frontier_only) - 1))
            row_sel = frontier_only.iloc[sel_idx]

            # In-sample prices (aligned)
            px_group_raw = _load_group_prices_cached(group)
            px_group_raw = px_group_raw.loc[:efb.END_DATE_INCLUSIVE].copy()
            px_group_raw = px_group_raw.dropna(axis=0, how="any")

            # References in-sample
            ref_disp = tuple(REFERENCE_DISPLAY_ORDER)
            ref_px = None
            try:
                ref_px = _fetch_reference_prices_in_sample(
                    display_tickers=ref_disp,
                    start=pd.Timestamp(px_group_raw.index.min()),
                    end_inclusive=pd.Timestamp(px_group_raw.index.max()),
                )
                ref_px = ref_px.dropna(axis=0, how="any")
            except Exception as e:
                st.warning(f"Could not load SPY/QQQ/DXY for the in-sample chart: {e}")

            # Common calendar across portfolio and refs
            if ref_px is not None and not ref_px.empty:
                common_idx = px_group_raw.index.intersection(ref_px.index)
                px_group = px_group_raw.loc[common_idx].copy()
                ref_px = ref_px.loc[common_idx].copy()
            else:
                px_group = px_group_raw

            # Portfolio weights: Selected + anchors
            w_sel = _weights_from_row(row_sel, tickers)

            anchors = {
                "Selected": w_sel,
            }

            gmv_row = frontier_df[frontier_df["point_type"].eq("GMV")].iloc[0]
            max_row = frontier_df[frontier_df["point_type"].eq("MaxReturn")].iloc[0]
            anchors["GMV"] = _weights_from_row(gmv_row, tickers)
            anchors["MaxReturn"] = _weights_from_row(max_row, tickers)

            s1_row = frontier_df[frontier_df["point_type"].eq("Slope1")]
            if not s1_row.empty:
                anchors["Slope1"] = _weights_from_row(s1_row.iloc[0], tickers)

            # Build in-sample base100 for portfolios
            port_series = {
                name: efb.base100_buy_and_hold(px_group, w)
                for name, w in anchors.items()
            }
            port_df = pd.concat(port_series, axis=1)

            # Add benchmark refs (base100)
            joined = port_df.copy()
            if ref_px is not None and not ref_px.empty:
                joined = pd.concat([joined, ref_px], axis=1).dropna(axis=0, how="any")
                for t in ref_disp:
                    if t in joined.columns:
                        joined[t] = efb.base100_single(joined[t])

            if joined.empty:
                st.error("Base 100 chart has no data after alignment.")
                st.stop()

            # Button that extends the base100 chart
            st.caption("Toggle any line on/off by clicking the legend.")
            if st.button("Live performance (extend chart)", use_container_width=True):
                st.session_state.show_live = True

            cutoff_actual = pd.Timestamp(joined.index.max())

            # Build figure (in-sample) + optional live extension
            base_fig = go.Figure()

            for col in joined.columns:
                base_fig.add_trace(
                    go.Scatter(
                        x=joined.index,
                        y=joined[col],
                        mode="lines",
                        name=str(col),
                        legendgroup=str(col),
                        showlegend=True,
                    )
                )

            if st.session_state.show_live:
                with st.spinner("Fetching live prices…"):
                    start_live = cutoff_actual

                    # Portfolio tickers and references together
                    ref_yf = [REFERENCE_DISPLAY_TO_YF.get(t, t) for t in ref_disp]
                    fetch_list = tuple(list(tickers) + ref_yf)
                    try:
                        px_live = _fetch_live_prices(fetch_list, start_inclusive=start_live)
                    except Exception as e:
                        st.error(f"Could not fetch live prices: {e}")
                        st.stop()

                px_live = px_live.copy()
                ren = {REFERENCE_DISPLAY_TO_YF.get(t, t): t for t in ref_disp}
                px_live = px_live.rename(columns=ren)
                px_live = px_live.dropna(axis=0, how="any")

                # Portfolio live segment (scaled to continue from in-sample)
                px_live_port = px_live[[t for t in tickers if t in px_live.columns]].copy()
                if px_live_port.empty:
                    st.warning("Live prices missing portfolio tickers; cannot extend chart.")
                else:
                    for name, w in anchors.items():
                        live_raw = efb.base100_buy_and_hold(px_live_port, w)
                        scale = float(joined.loc[cutoff_actual, name]) / 100.0 if name in joined.columns else 1.0
                        live_scaled = live_raw * scale
                        live_scaled = live_scaled.loc[live_scaled.index > cutoff_actual]
                        if live_scaled.empty:
                            continue

                        mode = "lines+markers" if name == "Selected" else "lines"
                        base_fig.add_trace(
                            go.Scatter(
                                x=live_scaled.index,
                                y=live_scaled.values,
                                mode=mode,
                                name=str(name),
                                legendgroup=str(name),
                                showlegend=False,
                                line=dict(dash="dot"),
                            )
                        )

                    # Benchmarks live segment
                    for t in ref_disp:
                        if t not in px_live.columns or t not in joined.columns:
                            continue
                        live_ref = efb.base100_single(px_live[t])
                        scale = float(joined.loc[cutoff_actual, t]) / 100.0
                        live_ref = (live_ref * scale).loc[live_ref.index > cutoff_actual]
                        if live_ref.empty:
                            continue
                        base_fig.add_trace(
                            go.Scatter(
                                x=live_ref.index,
                                y=live_ref.values,
                                mode="lines",
                                name=str(t),
                                legendgroup=str(t),
                                showlegend=False,
                                line=dict(dash="dot"),
                            )
                        )

            base_fig.update_layout(
                title=f"Base 100 (Buy & Hold, Adj Close) — {group}{subtitle}",
                xaxis_title="Date",
                yaxis_title="Index (Base 100)",
                margin=dict(l=10, r=10, t=60, b=10),
                legend=dict(itemclick="toggle", itemdoubleclick="toggleothers", groupclick="togglegroup"),
            )
            st.plotly_chart(base_fig, height=520, use_container_width=True)

            if st.session_state.show_live:
                st.caption(
                    "Live segment is shown as dotted lines (Selected also has markers) and extends the chart beyond the cutoff."
                )
            else:
                st.caption("Base 100 uses buy & hold with initial weights (no rebalancing).")


# Right panel: selected point details + anchors
with right:
    if st.session_state.selected_point_idx is None:
        st.markdown("### Selected portfolio")
        st.caption("Select a point on the frontier to see weights and metrics.")
    else:
        sel_idx = int(st.session_state.selected_point_idx)
        sel_idx = max(0, min(sel_idx, len(frontier_only) - 1))
        row_sel = frontier_only.iloc[sel_idx]
        _render_point(f"Selected point — #{sel_idx}", row_sel, tickers)

    st.divider()

    # Slope anchor
    s1 = frontier_df[frontier_df["point_type"].eq("Slope1")]
    if s1.empty:
        st.caption("No slope≈1 anchor found for this frontier.")
    else:
        _render_point("Slope≈1 anchor", s1.iloc[0], tickers)


# Bottom: GMV and MaxReturn in expanders
st.divider()

c1, c2 = st.columns(2, gap="large")

with c1:
    gmv = frontier_df[frontier_df["point_type"].eq("GMV")]
    if not gmv.empty:
        with st.expander("GMV (minimum variance)", expanded=True):
            _render_point("GMV (minimum variance)", gmv.iloc[0], tickers)

with c2:
    mx = frontier_df[frontier_df["point_type"].eq("MaxReturn")]
    if not mx.empty:
        with st.expander("Max return (long-only)", expanded=True):
            _render_point("Max return (long-only)", mx.iloc[0], tickers)


# Debug (optional)
with st.expander("Debug (frontier params)"):
    st.json(
        {
            "group": fr.group,
            "mu_method": fr.mu_method,
            "cap": fr.cap,
            "requested_targets": st.session_state.frontier_requested_points,
            "frontier_points_produced": st.session_state.frontier_front_points,
            "n_points_param": fr.n_points,
            "prices_start": str(fr.start_date.date()),
            "prices_end": str(fr.end_date.date()),
            "tickers_used": fr.tickers,
        }
    )
