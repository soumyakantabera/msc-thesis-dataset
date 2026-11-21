# When the Warehouse Fills Up: Inventory Buildup as an Early-Warning Signal

*MSc thesis: code, data and results (version 9)*

---

## The question

Most of us have seen a shop with too much stock: discount racks, half-empty
tills, staff with little to do. A year later it's often running clearance sales
or worse. This thesis asks whether the same pattern shows up in company
accounts, at scale and in a form you could test.

The question is simple. **When a company's inventory grows faster than its
sales, more than its peers' inventory does, is that an early warning of trouble
ahead?**

I measure this with **Abnormal Inventory Growth (AIG)**: inventory growth minus
revenue growth, compared with the median firm in the same industry group in the
same year. The peer adjustment matters. It stops the measure from flagging a
whole sector that builds stock ahead of a busy season, or every firm during the
2021–22 supply-chain shock. It picks out firms that are piling up goods *their
competitors are not*.

## The sample

- **279 companies** and **3,055 firm-years** of annual data, covering fiscal
  years 2014–2026 as reported in FactSet.
- They come from three index universes: the **Russell 1000** (US), the
  **FTSE** (UK) and the **STOXX** (continental Europe).
- They span 14 countries: USA 166, UK 38, France 19, Germany 18,
  Switzerland 10, Sweden 9, Italy 5, and smaller numbers from Finland, the
  Netherlands, Belgium, Denmark, Austria, Spain and Ireland.
- They are grouped into six peer "super-groups" whose inventory economics are
  alike:

| Group | Industries | Firms |
|---|---|---|
| S1 | Mass retail, home & specialty retail | 44 |
| S2 | Beverages, consumer staples, apparel | 66 |
| S3 | Industrials, automotive, aerospace | 71 |
| S4 | Tech hardware, semiconductors | 29 |
| S5 | Building materials, chemicals | 48 |
| S6 | Medical devices | 21 |

The screen started with 297 candidates and kept 279. Thirteen were dropped
because revenue was under $1bn after conversion to USD, three because inventory
was too small to matter, and two because they had too few years of history.
Every decision is logged, firm by firm, in
`ph1_raw_data/FINAL_screening_decisions.csv`.

One limitation should be stated up front: about **59% of the panel is
American**. This is best read as an *international extension of a US-anchored
sample*, not a truly global study, and several results below behave differently
inside and outside the US.

## What I found

I tested the idea through three economic channels. Inventory trouble should
show up in **margins**, in **cash** and in **stock returns**.

| | Hypothesis | Result | How to read it |
|---|---|---|---|
| **Margin** | H1: AIG predicts next year's gross-margin change | β = −0.0152, p = 0.014 | About −0.25 pp of gross margin per one-SD rise in AIG (SD = 0.161) |
| | H4: AIG predicts a gross-margin *drop* of over 2 pp | χ² = 8.04, p = 0.005 | 17% of high-stress firms vs 13% of low-stress firms |
| **Cash** | H2: AIG goes with weaker operating cash flow | β = −0.116, p < 0.001 | The strongest and most robust result |
| **Returns** | H3: high-AIG firms underperform | Q5−Q1 = −4.7% a year, t = −3.0 | Equal-weighted only |

On the whole, **the signal is real but modest.** Firms that build inventory
out of step with their peers do tend to see margins squeeze the next year,
weaker cash generation and lower returns. None of the effects is dramatic,
which is what you would expect from a variable that public accounts already
show to every analyst.

## Where the results are weaker

I've tried to treat my own results as a sceptical referee would. These are the
caveats I think matter most:

1. **There are three channels, not four.** H4 is defined from the same margin
   outcome as H1, so the two agree about 91% of the time. They are one finding
   measured two ways, not two separate confirmations.
2. **H2 is not a forecast.** The cash-flow result is same-year: AIG and weak
   cash flow happen together. It is the strongest *association* in the thesis,
   but it does not "warn early." Only the margin and return results look
   forward.
3. **H1 is borderline under multiple testing.** With four tests, a strict
   Bonferroni cutoff is 0.0125, and p = 0.014 just misses it. H2, H3 and H4
   pass.
4. **H1 is driven by firms outside the US.** Within the US alone it is weak
   (p = 0.19), and it is not significant before COVID.
5. **H3 is a small-firm effect.** The return spread holds when every firm
   counts equally but disappears when firms are weighted by market value.
6. **I rejected my original sector hypothesis.** I expected the signal to be
   concentrated in particular industries. It wasn't, in **0 of 13**
   specifications, and I report that as a rejection.
7. **IFRS vs US GAAP is only a proxy.** The data don't record which
   inventory-costing method each firm uses (LIFO vs FIFO), so the accounting
   split shows only what each regime *allows*.

The full audit trail, including errors I found and fixed along the way, is in
[`FINAL_AUDIT_MEMO_v9.md`](FINAL_AUDIT_MEMO_v9.md).

## How the repository is organised

The folders follow the order of the research, from raw data to final
robustness checks. Each phase keeps its own script together with the tables
and figures it produced.

```
ph1_raw_data/            Raw inputs as downloaded (FactSet fundamentals, prices, FX rates, screening log)
ph2_data_cleaning/       Builds the analysis panel from the raw files
    build_panel_ecb.py       canonical builder -> PANEL_thesis_vars.csv
    PANEL_thesis_vars.csv    the dataset every later phase uses
ph3_descriptive/         Who is in the sample: distributions, country mix, IFRS vs GAAP
ph4_core_regressions/    The main tests (H1–H4), with a placebo check
ph5_supergroup/          Does the result hold within each of the six peer groups?
ph6_sector_robustness/   The sector-concentration test (rejected)
ph7_deep_dives/          Firm case studies, for illustration only
ph8_robustness/          13 alternative specifications, IFRS split, placebo tests
ph9_stressscore/         A composite StressScore showing the distress mechanism
v9_common.py             Shared definitions used by phases 3–9
logs/                    Console output from the most recent full run
```

## Reproducing everything

You need Python 3.11+ and a few standard libraries:

```bash
pip install numpy pandas scipy matplotlib statsmodels
```

Then, from the repository root:

```bash
python ph2_data_cleaning/build_panel_ecb.py
```

```bash
for d in ph3_* ph4_* ph5_* ph6_* ph7_* ph8_* ph9_*; do python $d/*.py; done
```

Every script uses fixed random seeds, so a re-run gives the same numbers. In
the most recent run, the panel rebuilt from the raw files was **identical** to
the shipped dataset, and every headline figure above matched the audit memo.
The full pipeline takes a few minutes on a laptop.

*(The FX download script, `ph2_data_cleaning/ecb_fx_collection.py`, needs
internet access to the European Central Bank's API. You don't need to run it:
the exchange rates it collects are already saved in `ph1_raw_data/`.)*

---

Thank you for taking the time to look through this. I'd welcome any questions
or criticism, especially on the points in *Where the results are weaker*.
