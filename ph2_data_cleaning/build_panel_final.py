"""
PANEL BUILDER FROM RAW DATA (Thesis v9 - FINAL (USD-first screen fix))
============================================================
Builds the full analysis-ready panel starting ONLY from raw files.

Inputs (place in same folder or update RAW_DIR):
  - factset_data_collection_russell1000.csv
  - factset_data_collection_ftse.csv
  - factset_data_collection_stoxx.csv
  - factset_stock_prices_full.csv
  - fred_macro_fx_annual_final.csv

Outputs:
  - PANEL_full.csv
  - PANEL_thesis_vars.csv

Features:
  - Exact screening from Thesis Outline v9
  - Log growth differences for AIG (primary)
  - Simple % growth kept for robustness
  - SEK/DKK flagged (excluded from Size-based tests)
  - Lagged macro controls
  - Forward DVs only on consecutive years
  - StressScore with contemporaneous dCFOMgn
  - Winsorization at 1/99
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ================== CONFIG ==================
RAW_DIR = Path(__file__).resolve().parents[1] / 'ph1_raw_data'   # Change if needed
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'; OUTPUT_DIR.mkdir(exist_ok=True)
# ============================================

def main():
    print("=" * 65)
    print("BUILDING PANEL FROM RAW DATA (Thesis v9 - FINAL (USD-first screen fix))")
    print("=" * 65)

    # 1. Load raw fundamentals
    print("\n[1/7] Loading raw FactSet fundamentals...")
    files = [
        "factset_data_collection_russell1000.csv",
        "factset_data_collection_ftse.csv",
        "factset_data_collection_stoxx.csv"
    ]
    fund = pd.concat([pd.read_csv(RAW_DIR / f, low_memory=False) for f in files], ignore_index=True)
    print(f"      Loaded {len(fund):,} rows from {len(files)} files")

    # 2. Create ratios needed for screening
    print("\n[2/7] Creating ratios for screening...")
    fund['InvSales'] = fund['Inventory'] / fund['Revenue'].replace(0, np.nan)
    fund['DIO'] = 365 * fund['Inventory'] / fund['COGS'].replace(0, np.nan)

    # 3. Apply screening (from Thesis Outline v9)
    # FIX (v3): convert revenue to USD BEFORE the $1bn size floor, and count only
    # years with ACTUAL revenue (not empty rows). The v2 builder compared raw local-ccy
    # revenue to a USD floor, wrongly dropping 4 UK firms (Breedon, Watches of Switzerland,
    # Premier Foods, Senior) whose GBP revenue is >$1bn, and admitted thin-history firms.
    print("\n[3/7] Applying screening (relevance + size/history)  [USD-first fix]...")
    FX_SCREEN = {'USD':1.0,'GBP':1.27,'EUR':1.08,'CHF':1.12,'SEK':0.095,'DKK':0.145}
    fund['_rev_usd_screen'] = fund.apply(
        lambda r: r['Revenue'] * FX_SCREEN.get(r['ReportCcy'], 1.0)
        if pd.notna(r['Revenue']) else np.nan, axis=1)
    firm_stats = (fund.groupby('Ticker')
                  .agg(n_years=('Revenue', lambda s: s.notna().sum()),   # REAL revenue years
                       med_Revenue_USD=('_rev_usd_screen', 'median'),
                       med_InvSales=('InvSales', 'median'),
                       med_DIO=('DIO', 'median'))
                  .reset_index())

    relevance = (firm_stats['med_InvSales'] >= 0.03) | (firm_stats['med_DIO'] >= 20)
    size_history = (firm_stats['n_years'] >= 6) & (firm_stats['med_Revenue_USD'] >= 1000)

    keep_tickers = firm_stats.loc[relevance & size_history, 'Ticker'].unique()
    fund = fund[fund['Ticker'].isin(keep_tickers)].drop(columns=['_rev_usd_screen']).copy()
    print(f"      Retained {len(keep_tickers)} firms after screening")

    # 4. Load prices and merge
    print("\n[4/7] Loading prices and merging...")
    prices = pd.read_csv(RAW_DIR / "factset_stock_prices_full.csv", low_memory=False)
    prices = prices.rename(columns={'Cal Year': 'Year'})
    panel = fund.merge(
        prices[['Ticker', 'Year', '12m Return', 'Excess Return', 'RegIdx Excess Return', 'Market Cap ($M)']],
        on=['Ticker', 'Year'], how='left'
    )

    # Peer-relative excess return (PRIMARY H3 benchmark): firm 12m return minus the
    # leave-one-out mean 12m return of its SuperGroup that year (pooled globally).
    panel['_ret'] = pd.to_numeric(panel['12m Return'], errors='coerce')
    grp = panel.groupby(['SuperGroup', 'Year'])['_ret']
    _sum = grp.transform('sum'); _cnt = grp.transform('count')
    _loo_sum = _sum - panel['_ret'].fillna(0)
    _loo_n = _cnt - panel['_ret'].notna().astype(int)
    panel['peer_mean_sg'] = np.where(_loo_n > 0, _loo_sum / _loo_n.replace(0, np.nan), np.nan)
    panel['peer_excess_sg'] = panel['_ret'] - panel['peer_mean_sg']
    panel = panel.drop(columns=['_ret'])

    # 5. FRED FX + Lagged Macro
    print("\n[5/7] Adding FRED FX conversion and lagged macro...")
    fred = pd.read_csv(RAW_DIR / "fred_macro_fx_annual_final.csv")
    fw = fred.pivot(index='cal_year', columns='series_id', values='value').reset_index()

    # FX rates
    fx = fw[['cal_year', 'DEXUSEU', 'DEXUSUK', 'DEXSZUS']].copy()
    fx['USD_per_EUR'] = fx['DEXUSEU']
    fx['USD_per_GBP'] = fx['DEXUSUK']
    fx['USD_per_CHF'] = 1.0 / fx['DEXSZUS']
    fx = fx.sort_values('cal_year').ffill()

    def get_fx_rate(row):
        if row['ReportCcy'] == 'USD':
            return 1.0
        r = fx[fx['cal_year'] == row['Year']]
        if len(r) == 0:
            return np.nan
        r = r.iloc[0]
        return {'EUR': r['USD_per_EUR'], 'GBP': r['USD_per_GBP'], 'CHF': r['USD_per_CHF']}.get(row['ReportCcy'], np.nan)

    panel['fx_to_usd'] = panel.apply(get_fx_rate, axis=1)
    panel['usd_convertible'] = panel['fx_to_usd'].notna()
    panel['flag_no_fred_fx'] = panel['ReportCcy'].isin(['SEK', 'DKK'])

    # Convert to USD
    for col in ['Revenue', 'COGS', 'GrossProfit', 'Inventory', 'Assets', 'CFO']:
        if col in panel.columns:
            panel[f'{col}_USD_fred'] = np.where(
                panel['usd_convertible'], panel[col] * panel['fx_to_usd'], np.nan
            )

    # Lagged macro controls
    macro_cols = ['USREC', 'FEDFUNDS', 'DGS10', 'UNRATE', 'CPIAUCSL', 'GDP', 'INDPRO', 'M2SL']
    macro = fw[['cal_year'] + [c for c in macro_cols if c in fw.columns]].copy()
    macro = macro.sort_values('cal_year')

    for col in ['CPIAUCSL', 'GDP', 'INDPRO', 'M2SL']:
        if col in macro.columns:
            macro[col + '_growth'] = macro[col].pct_change() * 100

    macro_lag = macro.copy()
    macro_lag['Year'] = macro_lag['cal_year'] + 1
    rename_map = {
        'USREC': 'macro_USREC',
        'FEDFUNDS': 'macro_FEDFUNDS',
        'DGS10': 'macro_DGS10',
        'UNRATE': 'macro_UNRATE',
        'CPIAUCSL_growth': 'macro_CPI_infl',
        'GDP_growth': 'macro_GDP_growth',
        'INDPRO_growth': 'macro_INDPRO_growth',
        'M2SL_growth': 'macro_M2_growth'
    }
    macro_lag = macro_lag.rename(columns=rename_map)
    keep_macro = [v for v in rename_map.values() if v in macro_lag.columns]
    panel = panel.merge(macro_lag[['Year'] + keep_macro], on='Year', how='left')

    # 6. Build Thesis Variables
    print("\n[6/7] Building thesis variables (log growth AIG as primary)...")
    panel = panel.sort_values(['Ticker', 'Year']).reset_index(drop=True)

    # Fillna for retailers
    panel['Receivables'] = panel['Receivables'].fillna(0)

    # Ratios
    panel['DIO'] = 365 * panel['Inventory'] / panel['COGS'].replace(0, np.nan)
    panel['DSO'] = 365 * panel['Receivables'] / panel['Revenue'].replace(0, np.nan)
    panel['DPO'] = 365 * panel['Payables'] / panel['COGS'].replace(0, np.nan)
    panel['CCC'] = panel['DIO'] + panel['DSO'] - panel['DPO']
    panel['GrossMargin'] = panel['GrossProfit'] / panel['Revenue']
    panel['CFOMgn'] = panel['CFO'] / panel['Revenue']
    panel['DebtToAssets'] = panel['Debt'] / panel['Assets']
    panel['CapexInt'] = panel['Capex'] / panel['Revenue']
    panel['AssetTurn'] = panel['Revenue'] / panel['Assets']
    panel['Size'] = np.log(panel['Assets_USD_fred'].where(panel['Assets_USD_fred'] > 0))

    # QUALITY GATE (v3): remove degenerate firm-years where revenue is so small that
    # GrossMargin is non-finite or wildly out of range (e.g. Rivian pre-scale: GM = -1200%).
    # These corrupt winsorisation and FE regressions. Require |GrossMargin| <= 2 (i.e. -200%..+200%).
    panel['GrossMargin'] = panel['GrossMargin'].replace([np.inf, -np.inf], np.nan)
    _bad = panel['GrossMargin'].notna() & (panel['GrossMargin'].abs() > 2.0)
    if _bad.any():
        print(f"      Quality gate: nulling {int(_bad.sum())} degenerate firm-years (|GM|>200%)")
        panel.loc[_bad, ['GrossMargin','CFOMgn']] = np.nan
    # re-require >=6 finite-GM years per firm; drop firms that no longer qualify
    _gm_years = panel.groupby('Ticker')['GrossMargin'].transform(lambda s: s.notna().sum())
    _drop_firms = panel.loc[_gm_years < 6, 'Ticker'].unique()
    if len(_drop_firms):
        print(f"      Quality gate: dropping {len(_drop_firms)} firm(s) with <6 finite-GM years: {list(_drop_firms)}")
        panel = panel[~panel['Ticker'].isin(_drop_firms)].copy()

    # Lags
    g = panel.groupby('Ticker', group_keys=False)
    for col in ['Inventory', 'Revenue', 'DIO', 'CCC', 'CFOMgn']:
        panel[f'{col}_lag'] = g[col].shift(1)

    # === PRIMARY: LOG GROWTH (v9) ===
    panel['InvGrowth_log'] = np.log(panel['Inventory'] / panel['Inventory_lag'])
    panel['RevGrowth_log'] = np.log(panel['Revenue'] / panel['Revenue_lag'])
    panel['AbnInvRaw'] = panel['InvGrowth_log'] - panel['RevGrowth_log']

    # Robustness: simple %
    panel['InvGrowth'] = panel['Inventory'] / panel['Inventory_lag'] - 1
    panel['RevGrowth'] = panel['Revenue'] / panel['Revenue_lag'] - 1
    panel['AbnInvRaw_simple'] = panel['InvGrowth'] - panel['RevGrowth']

    # Peer median within SuperGroup × Year
    panel['peer_med_abninv'] = panel.groupby(['SuperGroup', 'Year'])['AbnInvRaw'].transform('median')
    panel['AbnInvPeer'] = panel['AbnInvRaw'] - panel['peer_med_abninv']
    panel['AIG'] = panel['AbnInvPeer']

    panel['peer_med_abninv_simple'] = panel.groupby(['SuperGroup', 'Year'])['AbnInvRaw_simple'].transform('median')
    panel['AbnInvPeer_simple'] = panel['AbnInvRaw_simple'] - panel['peer_med_abninv_simple']

    # Forward DVs (only consecutive years)
    panel['GM_fwd'] = g['GrossMargin'].shift(-1)
    panel['CFOMgn_fwd'] = g['CFOMgn'].shift(-1)
    panel['Year_fwd'] = g['Year'].shift(-1)
    consec = (panel['Year_fwd'] - panel['Year'] == 1)
    panel['dGM_fwd'] = np.where(consec, panel['GM_fwd'] - panel['GrossMargin'], np.nan)
    panel['dCFOMgn_fwd'] = np.where(consec, panel['CFOMgn_fwd'] - panel['CFOMgn'], np.nan)
    panel['GM_distress'] = np.where(panel['dGM_fwd'].notna(), (panel['dGM_fwd'] < -0.02).astype(float), np.nan)

    # StressScore (contemporaneous dCFOMgn)
    panel['dCFOMgn'] = panel['CFOMgn'] - panel['CFOMgn_lag']   # same-year change (H2 correct horizon)
    F1 = (panel['AbnInvRaw'] > 0).astype(float)
    F2 = ((panel['DIO'] - panel['DIO_lag']) > 0).astype(float)
    F3 = ((panel['CCC'] - panel['CCC_lag']) > 0).astype(float)
    F4 = ((panel['CFOMgn'] - panel['CFOMgn_lag']) < 0).astype(float)
    valid_ss = panel[['AbnInvRaw', 'DIO_lag', 'CCC_lag', 'CFOMgn_lag']].notna().all(axis=1)
    panel['StressScore'] = np.where(valid_ss, F1 + F2 + F3 + F4, np.nan)

    # Winsorization at 1/99
    def winsor(s):
        return s.clip(lower=s.quantile(0.01), upper=s.quantile(0.99))

    for c in ['AIG', 'AbnInvPeer', 'AbnInvRaw', 'AbnInvPeer_simple',
              'dGM_fwd', 'dCFOMgn_fwd', 'GrossMargin', 'CFOMgn', 'DIO', 'CCC',
              'DebtToAssets', 'CapexInt', 'AssetTurn', 'Size']:
        if c in panel.columns:
            panel[f'{c}_w'] = winsor(pd.to_numeric(panel[c], errors='coerce'))

    # 7. Save outputs
    # Clean-up: drop rows with no valid Year (pre-IPO/spinoff empties where FactSet returned
    # no FY-end date). They carry no data, can't enter any test, and create phantom duplicates.
    _before = len(panel)
    panel = panel[panel['Year'].notna()].copy()
    panel['Year'] = panel['Year'].astype(int)
    print(f"      Dropped {_before - len(panel)} empty (NaN-Year) rows")
    print("\n[7/7] Saving output files...")
    panel.to_csv(OUTPUT_DIR / "PANEL_full.csv", index=False)
    panel.to_csv(OUTPUT_DIR / "PANEL_thesis_vars.csv", index=False)

    # Final Summary
    print("\n" + "=" * 65)
    print("PANEL BUILD COMPLETE - EXCELLENT STATUS")
    print("=" * 65)
    print(f"Total firm-years after screening : {len(panel):,}")
    print(f"Unique firms                     : {panel['Ticker'].nunique()}")
    print(f"AIG non-null                     : {panel['AIG'].notna().sum():,}")
    print(f"dGM_fwd non-null (usable H1/H2)  : {(panel['AIG'].notna() & panel['dGM_fwd'].notna()).sum():,}")
    print(f"12m Return non-null (usable H3)  : {(panel['AIG'].notna() & panel['12m Return'].notna()).sum():,}")
    print(f"StressScore non-null (usable H5) : {(panel['StressScore'].notna() & panel['GM_distress'].notna()).sum():,}")
    print(f"SEK/DKK flagged firms            : {panel['flag_no_fred_fx'].sum() if 'flag_no_fred_fx' in panel.columns else 'N/A'}")
    print("=" * 65)

if __name__ == "__main__":
    main()