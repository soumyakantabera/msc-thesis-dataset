# Thesis v9 — Source Package (phase-ordered: ph1 → ph9)

Inventory Buildup as an Early-Warning Signal — 279-firm international panel.
Folders follow the project's phase order so each phase's Python outputs
(tables/ and figures/) can be stored INSIDE its own phase folder.

## Folder structure & run order

ph1_raw_data/            Raw inputs (run nothing here)
  - factset_data_collection_russell1000 / _ftse / _stoxx .csv   raw fundamentals
  - factset_stock_prices_full.csv                               raw 12m returns
  - ecb_fx_annual.csv (canonical FX) · fred_macro_fx_annual_final.csv (predecessor)
  - FINAL_screening_decisions.csv                               sample-funnel log

ph2_data_cleaning/       Cleaning pipeline + THE PANEL
  1. ecb_fx_collection.py      -> ecb_fx_annual.csv
  2. build_panel_ecb.py        -> PANEL_thesis_vars.csv   [CANONICAL builder]
  -  build_panel_final.py      (FRED-FX predecessor, provenance only)
  -  PANEL_thesis_vars.csv     THE analysis dataset (279 firms, 3,055 firm-years)

ph3_descriptive/         descriptive_analysis_v9.py      (descriptives, IFRS panel, US/intl)
ph4_core_regressions/    phase4_regressions_v9.py        (H1–H4, placebo, 3-channel figure)
ph5_supergroup/          phase5_supergroup_v9.py         (S1–S6 decomposition, fragility flags)
ph6_sector_robustness/   phase6_sector_robustness_v9.py  (former-H4 sector test: REJECTED)
ph7_deep_dives/          phase7_deep_dives_v9.py         (firm cases, illustrative only)
ph8_robustness/          phase8_robustness_v9.py         (13-way rejection battery, IFRS split, placebo)
ph9_stressscore/         phase9_stressscore_v9.py        (StressScore distress mechanism)

Root:
  v9_common.py            REQUIRED shared module imported by ALL ph3–ph9 scripts
                          (SD=0.161, US-share, IFRS/GAAP capability flag)
  FINAL_AUDIT_MEMO_v9.md  audit trail + all honesty disclosures

## How to run
Paths are now relative to this package (no edits needed). From the package root:

    pip install numpy pandas scipy matplotlib statsmodels requests
    python ph2_data_cleaning/build_panel_ecb.py      # rebuild -> ph2_data_cleaning/outputs/ (verified identical to shipped panel)
    python v9_common.py
    for d in ph3_* ph4_* ph5_* ph6_* ph7_* ph8_* ph9_*; do python $d/*.py; done

Each phase writes tables/ and figures/ into its own folder; console logs are in logs/.
ecb_fx_collection.py needs internet access to the ECB API (writes ph2_data_cleaning/outputs/);
the panel build uses the shipped ph1_raw_data/ecb_fx_annual.csv instead.

## Storing outputs (the intended workflow)
Run each phase and save its tables/ + figures/ inside the matching phN_ folder, e.g.:
  ph4_core_regressions/
      phase4_regressions_v9.py
      tables/    <- phase 4 CSVs
      figures/   <- phase 4 PNGs

## Audited headline numbers (cross-phase consistent)
H1 ΔGM: β=−0.0152, p=0.014 (−0.245pp per 1-SD, SD=0.161) · H2 ΔCFO: β=−0.116, p<0.001
H3: Q5−Q1 EW −4.7%, t=−3.0 (VW insignificant) · H4: χ²=8.04, p=0.005
Sector (former H4): REJECTED, 0/13 specifications.
Structure: 3 economic channels (MARGIN H1+H4 · CASH H2 · RETURNS H3), not 4 pillars.
