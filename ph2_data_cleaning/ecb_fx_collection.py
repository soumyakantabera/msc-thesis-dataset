#!/usr/bin/env python3
"""
ECB DATA PORTAL — FX COLLECTION for the 279-firm international panel
====================================================================
Pulls the 5 non-USD currencies the panel needs (GBP, CHF, SEK, DKK + EUR itself)
and converts everything to USD-per-1-unit-of-local-currency.

WHY ECB: free, no API key, official central-bank reference rates, and — unlike FRED —
it covers SEK and DKK, which lets the 11 Nordic firms re-enter the USD-magnitude tests.

API (verified June 2026):
  base : https://data-api.ecb.europa.eu/service/data/EXR/
  key  : {FREQ}.{CCY}.EUR.SP00.{COLL}    (all ECB rates are quoted PER EURO)
         FREQ : 'A' annual, 'M' monthly, 'D' daily
         COLL : 'A' = period AVERAGE (use for FLOW items: Revenue, COGS, CFO, Capex...)
                'E' = END-OF-PERIOD  (use for STOCK items: Inventory, Assets, Debt...)
  fmt  : ?format=csvdata
  e.g. https://data-api.ecb.europa.eu/service/data/EXR/A.USD.EUR.SP00.A?format=csvdata

CONVERSION LOGIC (important):
  ECB gives  CCY-per-EUR  (e.g. USD/EUR = 1.08, GBP/EUR = 0.85, SEK/EUR = 11.2).
  We want    USD-per-CCY  to convert a firm's local figure into USD.
    USD_per_EUR = (USD/EUR)                      # direct
    USD_per_CCY = (USD/EUR) / (CCY/EUR)          # triangulate via EUR
  So: firm_value_local * USD_per_CCY = firm_value_USD.

OUTPUTS:
  ecb_fx_annual.csv  — one row per (year, currency) with USD_per_unit at BOTH
                       average (flows) and end-of-period (stocks) collection types.
"""
import io
import time
import sys
import requests
import pandas as pd

BASE = "https://data-api.ecb.europa.eu/service/data/EXR"
START, END = "2013-01-01", "2026-12-31"   # cover the panel period + 1 lag year

# Currencies the panel needs (ECB quotes each PER EURO). USD is the numeraire.
# We pull USD/EUR (to anchor) plus the 4 others; EUR firms use USD/EUR directly.
TARGETS = ["USD", "GBP", "CHF", "SEK", "DKK"]
# Collection types: A = period average (flows), E = end-of-period (stocks/balance sheet)
COLLECTIONS = {"avg": "A", "eop": "E"}


def fetch_series(ccy: str, coll: str, freq: str = "A", retries: int = 4) -> pd.DataFrame:
    """Fetch one ECB EXR series as a tidy DataFrame [year, value] (CCY per EUR)."""
    key = f"{freq}.{ccy}.EUR.SP00.{coll}"
    url = f"{BASE}/{key}"
    params = {"startPeriod": START, "endPeriod": END, "format": "csvdata"}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params,
                             headers={"Accept": "text/csv"}, timeout=30)
            if r.status_code == 200 and r.text.strip():
                df = pd.read_csv(io.StringIO(r.text))
                # columns include TIME_PERIOD and OBS_VALUE
                out = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
                out.columns = ["period", "ccy_per_eur"]
                # annual periods are 'YYYY'
                out["year"] = out["period"].astype(str).str[:4].astype(int)
                out["currency"] = ccy
                return out[["year", "currency", "ccy_per_eur"]]
            else:
                print(f"   [{key}] HTTP {r.status_code}, attempt {attempt}")
        except Exception as e:
            print(f"   [{key}] error attempt {attempt}: {e}")
        time.sleep(2 * attempt)
    print(f"   [{key}] FAILED after {retries} attempts")
    return pd.DataFrame(columns=["year", "currency", "ccy_per_eur"])


def collect():
    rows = []
    for label, coll in COLLECTIONS.items():
        print(f"\n[{label}] collection='{coll}' (annual)")
        per_eur = {}
        for ccy in TARGETS:
            df = fetch_series(ccy, coll, freq="A")
            if df.empty:
                print(f"   {ccy}: NO DATA")
                continue
            print(f"   {ccy}: {df['year'].min()}–{df['year'].max()} ({len(df)} obs)")
            per_eur[ccy] = df.set_index("year")["ccy_per_eur"]
            time.sleep(0.5)

        if "USD" not in per_eur:
            print("   FATAL: USD/EUR not retrieved; cannot triangulate.")
            sys.exit(1)

        usd_per_eur = per_eur["USD"]  # USD per 1 EUR
        for year in usd_per_eur.index:
            # EUR itself: USD per EUR = USD/EUR
            rows.append((year, "EUR", label, float(usd_per_eur[year])))
            # USD: trivially 1
            rows.append((year, "USD", label, 1.0))
            # others: USD_per_CCY = (USD/EUR) / (CCY/EUR)
            for ccy in ["GBP", "CHF", "SEK", "DKK"]:
                s = per_eur.get(ccy)
                if s is not None and year in s.index and s[year] != 0:
                    rows.append((year, ccy, label, float(usd_per_eur[year]) / float(s[year])))

    out = pd.DataFrame(rows, columns=["year", "currency", "collection", "usd_per_unit"])
    # pivot so each currency-year has both avg and eop columns
    wide = out.pivot_table(index=["year", "currency"], columns="collection",
                           values="usd_per_unit").reset_index()
    wide = wide.rename(columns={"avg": "USD_per_unit_avg", "eop": "USD_per_unit_eop"})
    wide = wide.sort_values(["currency", "year"]).reset_index(drop=True)

    # --- 2026 fallback: annual ECB data ends at the last closed year. For fiscal years
    # labelled 2026, pull the LATEST available MONTHLY rate per currency and append as 2026.
    max_year = int(wide["year"].max())
    if max_year < 2026:
        print(f"\n[2026 fallback] annual ends {max_year}; pulling latest monthly rate for 2026...")
        per_eur_m = {}
        for ccy in TARGETS:
            for coll in ["A", "E"]:
                m = fetch_series(ccy, coll, freq="M")
                if not m.empty:
                    m26 = m[m["year"] == 2026]
                    src = m26 if len(m26) else m
                    per_eur_m[(ccy, coll)] = float(src["ccy_per_eur"].iloc[-1])
                time.sleep(0.3)
        add = []
        for label, coll in COLLECTIONS.items():
            ue = per_eur_m.get(("USD", coll))
            if ue is None:
                continue
            add.append((2026, "EUR", label, ue))
            add.append((2026, "USD", label, 1.0))
            for ccy in ["GBP", "CHF", "SEK", "DKK"]:
                ce = per_eur_m.get((ccy, coll))
                if ce:
                    add.append((2026, ccy, label, ue / ce))
        addw = (pd.DataFrame(add, columns=["year", "currency", "collection", "usd_per_unit"])
                .pivot_table(index=["year", "currency"], columns="collection", values="usd_per_unit")
                .reset_index().rename(columns={"avg": "USD_per_unit_avg", "eop": "USD_per_unit_eop"}))
        wide = pd.concat([wide, addw], ignore_index=True).sort_values(["currency", "year"]).reset_index(drop=True)
        print(f"   appended 2026 rows: {len(addw)}")

    __import__("os").makedirs(__import__("os").path.join(__import__("os").path.dirname(__file__),"outputs"),exist_ok=True); wide.to_csv(__import__("os").path.join(__import__("os").path.dirname(__file__),"outputs","ecb_fx_annual.csv"), index=False)

    print("\n" + "=" * 60)
    print("ECB FX COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Rows: {len(wide)} | currencies: {sorted(wide['currency'].unique())}")
    print(f"Years: {wide['year'].min()}–{wide['year'].max()}")
    print("\nSample (use _avg for flows, _eop for balance-sheet items):")
    print(wide[wide["currency"].isin(["GBP", "SEK", "DKK"])]
          .query("year in [2020, 2024]").to_string(index=False))
    return wide


if __name__ == "__main__":
    collect()
