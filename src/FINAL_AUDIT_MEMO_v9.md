# Final Audit Memo — Thesis v9 Analysis (Phases 3–9)

Adversarial, pessimistic, deterministic review before write-up. All six phases
run clean; all cross-phase numbers reconcile; all findings tables grounded in
verified output.

## Cross-phase consistency — PASSES
| Statistic | Phase A | Phase B | Match |
|---|---|---|---|
| H1 headline β | phase4 −0.0152 | phase8 alt-IV −0.0152 | yes |
| H4 distress χ² | phase4 8.04 | phase9 8.04 | yes |
| Semis β (rejection) | phase6 −0.0197 | phase8 −0.0197 | yes |
| AIG SD (per-SD basis) | 0.161 everywhere | — | yes |

No number contradicts another across phases.

## Issues the final audit CAUGHT and FIXED (disclosure, not bugs)
1. **H1 and H4 are NOT independent evidence (CRITICAL).** GM_distress = 1[ΔGM<−2pp];
   H4 is a binary recoding of H1's outcome (~91% identical full-panel; 100% on the
   H4 subsample). → There are **3 distinct economic channels, not 4**: MARGIN (H1=H4),
   CASH (H2), RETURNS (H3). Added to phase4 findings.
2. **H2 is contemporaneous, not predictive.** Same-year ΔCFO (t), while the thesis is
   titled "early-warning." H2 is the strongest *association* but a concurrent cash-drain
   signal, NOT a forecast. Only H1/H3/H4 are forward (t+1). Reframed in phase4 findings.
3. **H1 marginal under multiple testing.** p=0.014 fails strict Bonferroni (0.0125 at
   4 tests); H2/H3/H4 pass. Added to phase4 findings.
4. **Phase 9 not independent of H1.** corr(StressScore, AbnInvPeer)=0.62 AND shares the
   margin outcome → phase 9 is mechanism illustration, not separate corroboration.
   Added to phase9 findings.

## Issues caught in earlier per-phase audits (already fixed)
- ph6: significant sectors were interaction-based (mislabelled Mass Retail sig, hid
  Staples) → switched to standalone FE: Medical Devices + Consumer Staples sig at 5%,
  Mass Retail borderline (p=0.058).
- ph4: H4 falsely called "holds in both US/intl" → corrected (NOT sig ex-US, χ²=0.56).
- ph3: Table 5 (US/intl) and Table 6 (GAAP/IFRS) are the SAME partition → reframed.
- ph5: H2 "all groups" → "negative in all 6, sig at 5% in 4, at 10% in all 6";
  H1 per-group sig rests on FRAGILE Healthcare → flagged, pooled is reportable.
- ph9: continuous-ΔGM finding REVERSED from v8 null → now sig (−0.56pp, p<0.001);
  monotonic-plateau not "noise"; zero SGs sig at 5%; F1–F4 dropped (not in panel).

## Standing honesty flags for the write-up
- Panel 59% US — "international extension of a US-anchored sample," not global.
- H1 conditional: weak US-only (p=0.19), stronger ex-US; not sig pre-COVID.
- H3 significance equal-weighted only; value-weighted insignificant (smaller-firm effect).
- H2 horizon + sector demotion chosen post-hoc — labelled.
- Sector hypothesis (former H4) REJECTED across 13 specifications — reported as rejection.
- IFRS/GAAP is a capability proxy (no observed inventory-method field).

## Verdict
The four-hypothesis structure is sound but should be presented as **3 channels**
(margin, cash, returns) with H2 as a contemporaneous association and the margin
channel evidenced two ways (H1 continuous + H4 binary). H2 is the one universally
robust result; H1/H3/H4 are real but conditional; sector concentration is rejected.
No fabrication anywhere; every magnitude on SD=0.161; every claim traces to an
audited script + findings table.

---
## LAST AUDIT PASS (final checkup, ph3–ph9)

**Verified clean:**
- All 7 phase scripts run end-to-end without error (ph7 now included).
- DETERMINISM verified by double-run + byte-diff: identical outputs, including the
  seeded placebo (phase 4) and — after fix — the phase 9 jitter scatter.
- Cross-phase reconciliation re-verified post-regeneration: H1 β, H4 χ², semis β all match.
- Every quantitative findings-table claim spot-verified against the data
  (ph6 sector p-values, ph9 distress gradient, ph4 channel agreement, ph8 13-spec count).
- All 7 dashboards embed 100% of produced figures and carry findings tables.
- All disclosure rows present and consistent with this memo.

**Issues CAUGHT and FIXED in this last pass:**
1. Phase 9 jitter scatter used unseeded RNG → non-deterministic figure. Seeded (42).
2. 12 stale files from pre-rebuild runs had survived in deliverables
   (10 old-numbering tables + 2 figures in v9_results; Figure23/26 in sector folder;
   TableUS_share ×2 in descriptive; Table19_stress in supergroup). All purged from
   BOTH work dir and deliverables; deliverables now byte-identical to fresh outputs.
3. Deep-dive dashboard silently omitted Figure34 (Tesla timeline). Added; all
   dashboards now embed every produced figure.

**Final state: ZERO outstanding errors.** Scripts deterministic, outputs fresh,
claims verified, disclosures consistent. Analysis is closed for write-up.
