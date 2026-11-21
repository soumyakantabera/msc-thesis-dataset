#!/usr/bin/env python3
"""
v9_common.py — shared helpers for cross-cutting v9 fixes
========================================================
Three audit-driven fixes used across phases 3–8 so every script reports them
identically (one source of truth, no drift):

  FIX 1  per-SD economic magnitude — uses the ESTIMATION-SAMPLE SD of AbnInvPeer
         (0.161), not the v8 value (0.118). A "1-SD" statement must use the SD of
         the sample the coefficient was estimated on.
  FIX 2  US-share disclosure — every cut (overall, super-group, sector) carries its
         US firm share, so the analysis owns its US dominance (59%).
  FIX 3  IFRS/GAAP capability flag — the panel has NO actual inventory-method field,
         so we can only flag LIFO-CAPABILITY by accounting regime (US-GAAP permits
         LIFO; IFRS bans it). This is a capability proxy, NOT an observed-method flag.
         Any thesis text must state this caveat.

Import and call from each phase script:
    from v9_common import aig_sd, per_sd, us_share, add_regime_flag, US_NOTE
"""
import numpy as np, pandas as pd

# ---------------------------------------------------------------------------
# FIX 1 — economic magnitude on the correct SD
# ---------------------------------------------------------------------------
def aig_sd(panel: pd.DataFrame, iv: str = "AbnInvPeer",
           dv: str = "dGM_fwd", controls=("Size_w", "DebtToAssets_w", "RevGrowth_w")) -> float:
    """SD of the IV on the H1 ESTIMATION sample (non-missing dv, iv, controls).
    This is the SD to use for per-1-SD economic-magnitude statements."""
    cols = [iv, dv] + [c for c in controls if c in panel.columns]
    s = panel[cols].apply(pd.to_numeric, errors="coerce").dropna()
    return float(s[iv].std())

def per_sd(beta: float, sd: float, pct_points: bool = True) -> float:
    """Translate a coefficient into the effect of a 1-SD increase in the IV.
    pct_points=True returns the answer in percentage points (×100)."""
    val = beta * sd
    return round(val * 100, 3) if pct_points else round(val, 5)

# ---------------------------------------------------------------------------
# FIX 2 — US-share disclosure
# ---------------------------------------------------------------------------
def us_share(panel: pd.DataFrame, by: str | None = None,
             firm: str = "Ticker", country: str = "Country", us_label: str = "USA") -> pd.DataFrame:
    """Return US firm-share overall (by=None) or by a grouping column (e.g. 'SuperGroup',
    'Sector'). Columns: [group,] firms, us_firms, us_share_pct."""
    def _one(df):
        tot = df[firm].nunique()
        us = df[df[country] == us_label][firm].nunique()
        return pd.Series({"firms": tot, "us_firms": us,
                          "us_share_pct": round(100 * us / tot, 1) if tot else np.nan})
    if by is None:
        out = _one(panel).to_frame().T
        out.insert(0, "group", "ALL")
        return out.reset_index(drop=True)
    rows = [pd.concat([pd.Series({by: g}), _one(d)]) for g, d in panel.groupby(by)]
    return pd.DataFrame(rows).reset_index(drop=True)

US_NOTE = ("Panel is US-anchored: 166 of 279 firms (59%) are US-domiciled. "
           "Results are an international extension of a US-centred sample, not a "
           "balanced global panel; non-US cells in some sectors are thin.")

# ---------------------------------------------------------------------------
# FIX 3 — IFRS/GAAP accounting-regime capability flag (NOT observed method)
# ---------------------------------------------------------------------------
def add_regime_flag(panel: pd.DataFrame, country: str = "Country", us_label: str = "USA") -> pd.DataFrame:
    """Add two columns:
        acct_regime    : 'US-GAAP' for US firms, 'IFRS' otherwise
        lifo_capable   : 1 if the regime PERMITS LIFO (US-GAAP), else 0
    IMPORTANT: this is a *capability* proxy by regime. The panel has no observed
    inventory-method field, so we cannot say a given US firm actually uses LIFO —
    only that it could. Cross-country DIO comparisons should carry this caveat."""
    df = panel.copy()
    df["acct_regime"] = np.where(df[country] == us_label, "US-GAAP", "IFRS")
    df["lifo_capable"] = (df["acct_regime"] == "US-GAAP").astype(int)
    return df

IFRS_NOTE = ("Accounting-regime flag is a capability proxy: US firms report under US-GAAP "
             "(LIFO permitted), non-US firms under IFRS (LIFO prohibited). The dataset does "
             "not record the actual inventory-costing method, so 'lifo_capable' marks only "
             "what the regime allows. Under LIFO, reported inventory and DIO can be understated "
             "versus FIFO/IFRS peers, so cross-country DIO levels are not perfectly comparable.")

# convenience: the canonical numbers, computed once if run directly
if __name__ == "__main__":
    p = pd.read_csv(__import__("pathlib").Path(__file__).resolve().parent/"ph2_data_cleaning"/"PANEL_thesis_vars.csv", low_memory=False)
    sd = aig_sd(p)
    print(f"FIX1  estimation-sample AIG SD = {sd:.4f}  (v8 was 0.118; ratio {sd/0.118:.2f}x)")
    print(f"      example: beta -0.0152 -> {per_sd(-0.0152, sd):+.3f} pp per 1-SD")
    print("\nFIX2  US-share overall:")
    print(us_share(p).to_string(index=False))
    print("\n      US-share by super-group:")
    print(us_share(p, by="SuperGroup").to_string(index=False))
    pf = add_regime_flag(p)
    print("\nFIX3  regime counts (firm-years):")
    print(pf.groupby("acct_regime")["Ticker"].nunique().to_string())
    print("\n" + IFRS_NOTE)
