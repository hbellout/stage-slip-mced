# Galleri Stage Slip Simulation

**Why the NHS-Galleri trial missed its primary endpoint — and why it matters.**

## The Finding

The NHS-Galleri trial (142,000 participants, England, 2021–2025) reported a ~30% reduction in Stage IV cancer diagnoses using the Galleri multi-cancer blood test, but failed to meet its primary endpoint: a reduction in combined Stage III+IV diagnoses.

This repository contains a Monte Carlo simulation that explains why. The Galleri test detected approximately **84 cancers at Stage I/II** that were subsequently recorded as Stage III+IV because the NHS diagnostic pathway took too long to confirm them. The trial needed just **25 of these 84 cases** (one in three) to have been diagnosed promptly for the primary endpoint to reach statistical significance.

The endpoint was not missed because the test failed. It was missed because the diagnostic infrastructure was not fast enough to keep up with what the test found.

## Repository Contents

```
├── simulate.py              # Core Monte Carlo simulation (v3, calibrated)
├── analyse_slip.py          # Intervention-arm slip analysis and p-value sensitivity
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── docs/
│   ├── executive_summary.tex    # 2-page general-audience summary
│   └── technical_narrative.tex  # Full technical document with methodology
└── output/                  # Generated LaTeX tables (created by simulate.py)
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the full simulation: calibration + 200-seed run + delay sweep
# Takes ~5 minutes. Produces LaTeX tables in output/
python simulate.py

# Run the slip analysis: intervention-arm case counts + p-value sensitivity
python analyse_slip.py
```

## How It Works

### Simulation Architecture

The simulation tracks virtual cancer cases through onset → detection → diagnosis using an event-time Monte Carlo framework:

- **12 cancer types** weighted to English population incidence
- **Cancer-specific MCED sensitivity** from the CCGA 3 clinical validation study (Klein 2021, Liu 2020)
- **Tumour progression** modelled as lognormal sojourn times per stage
- **SoC referral** via stage-dependent exponential hazards
- **Diagnostic delay** modelled as lognormal with configurable median (central: 92 days)
- **Three annual screening rounds** with a learning curve on workup times

### Calibration

Per-cancer referral hazards are calibrated against NHS England population stage-at-diagnosis distributions compiled from:

- National Lung Cancer Audit (NLCA) 2022–2023
- NHS Digital cancer incidence by stage (2015–2022)
- Cancer Research UK Early Diagnosis Data Hub
- Nuffield Trust cancer survival statistics

Post-calibration, all 12 cancers match NHS population data within 1–2 percentage points RMS across all stage bins.

### The 92-Day Delay Parameter

The SoC diagnostic delay median of 92 days is a constructed estimate derived from five convergent evidence streams:

1. **PATHFINDER 2** (US optimised floor): 46-day median resolution
2. **PATHFINDER 1** (early US experience): 79-day median, 162-day false-positive median
3. **Mann et al. (RAND, 2025)**: +3.4pp increase in NHS diagnostic delays in trial regions
4. **GRAIL SEC disclosure (Feb 2026)**: confirms "higher than anticipated Stage III" linked to "time to diagnostic resolution"
5. **NHS system data**: 62-day standard met for only 67–72% of patients; 52.3% of urgent referrals exceeded 28-day target

Construction: 46 days (PATHFINDER 2 floor) × 2.0 (NHS multiplier) = 92 days.

The conclusion is robust across the full 65–120 day range: at every delay value tested, the number of slipped cases exceeds the 25-case significance threshold.

## Key Output

From `analyse_slip.py` (200 seeds, calibrated model, CCGA-3 cancer-specific sensitivities):

| Metric | Value |
|--------|-------|
| Control Stage III+IV | 470 [434, 505] |
| Intervention Stage III+IV | 443 [407, 479] |
| **Intervention slip (E→III+IV)** | **84 [68, 104]** |
| — Screen route (test+, workup too slow) | 42 [28, 55] |
| — SoC route (test−, standard progression) | 43 [30, 55] |
| System had opportunity (86%) | 73 [57, 88] |
| Cases to recover for p<0.05 | ~25 (32% of slip) |

The slip count is robust: it remains between 74 and 85 across a 3× range of sensitivity assumptions and across the full 65–120 day delay range.

## Reproducibility

All results are deterministic. Seeds are integers 0–199 for the control arm and 100,000–100,199 for the intervention arm, using `numpy.random.default_rng()`. Running the scripts with the same NumPy version produces identical output.

Tested with Python 3.10+ and NumPy 1.24+.

## Documents

- **`docs/executive_summary.tex`** — 2-page general-audience summary. Leads with the finding, then builds credibility.
- **`docs/technical_narrative.tex`** — Full methodology: model mechanics, calibration verification, 92-day parameter derivation, simulation outputs, limitations, and scope of claims.

Compile with `pdflatex`.

## Citation

If you use this work, please cite:

```
Stage Slip in MCED Trials: A Calibrated Monte Carlo Simulation
of the NHS-Galleri Trial. February 2026.
https://github.com/[your-username]/galleri-stage-slip
```

## License

MIT
