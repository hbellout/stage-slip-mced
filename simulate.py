#!/usr/bin/env python3
"""
NHS_Galleri_12CancerDelaySim_v3_calibrated.py

v3: Adds calibration of per-cancer referral hazards against NHS/CRUK
population stage distributions. Then runs the full simulation with
calibrated parameters.

Calibration method:
  For each cancer, we run the control arm repeatedly while adjusting
  (kE, kIII, kIV) to minimise the squared gap between simulated and
  target stage distributions. We use a simple grid search over scale
  factors applied to a base hazard structure, keeping the ordering
  constraint kE < kIII < kIV.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
import os, copy

# ============================================================================
# CONFIG (same as v2)
# ============================================================================

@dataclass
class Config:
    N_screened: int = 142_000
    incidence_frac: float = 0.010
    T_follow: float = 3.0
    T_pre: float = 2.0
    screen_times: list = field(default_factory=lambda: [0.0, 1.0, 2.0])
    delay_sigma: float = 0.55
    sojourn_sigma: float = 0.35
    soc_delay_median_days: float = 92.0
    screen_delay_median_days: list = field(default_factory=lambda: [75.0, 60.0, 50.0])
    enable_spillover: bool = True
    spillover_factor: float = 1.20
    spillover_t_end: float = 1.0
    enable_learning: bool = True
    enable_upstaging: bool = False
    upstage_prob_screen: float = 0.06
    upstage_prob_soc: float = 0.03
    n_seeds: int = 200


@dataclass
class Cancer:
    name: str
    weight: float
    medE: float
    medIII: float
    medIV: float
    medSurvIV: float
    sensE: float
    sensIII: float
    sensIV: float
    kE: float = 0.25
    kIII: float = 1.00
    kIV: float = 2.50


# ============================================================================
# CANCER TABLE with CCGA 3 sensitivities (same as v2)
# ============================================================================

def cancer_table_12() -> List[Cancer]:
    return [
        # Sensitivities from published CCGA-3 cancer-specific data (Klein 2021,
        # GRAIL HCP FAQ, Shao 2022). These 12 cancers have substantially higher
        # sensitivity than the all-cancer averages because they are aggressive
        # tumours that shed more cfDNA at earlier stages.
        Cancer("Anus",       0.020, 1.2, 0.7, 0.5, 0.9,  0.45, 0.75, 0.90),
        Cancer("Bladder",    0.080, 1.2, 0.6, 0.5, 1.0,  0.17, 0.50, 0.75),
        Cancer("Colorectal", 0.200, 2.5, 0.9, 0.6, 1.2,  0.30, 0.70, 0.90),
        Cancer("Esophagus",  0.050, 0.9, 0.5, 0.4, 0.8,  0.40, 0.80, 0.92),
        Cancer("Head_Neck",  0.050, 1.1, 0.6, 0.5, 1.0,  0.50, 0.85, 0.95),
        Cancer("Liver_Bile", 0.040, 0.8, 0.4, 0.3, 0.7,  0.80, 1.00, 1.00),
        Cancer("Lung",       0.230, 0.8, 0.5, 0.3, 0.7,  0.40, 0.84, 0.93),
        Cancer("Lymphoma",   0.090, 1.6, 0.8, 0.7, 1.2,  0.40, 0.73, 0.88),
        Cancer("Myeloma",    0.060, 1.8, 0.9, 0.7, 1.3,  0.35, 0.70, 0.85),
        Cancer("Ovary",      0.060, 1.0, 0.6, 0.5, 1.0,  0.65, 0.87, 0.95),
        Cancer("Pancreas",   0.060, 0.5, 0.3, 0.25, 0.5, 0.61, 0.86, 0.96),
        Cancer("Stomach",    0.060, 1.0, 0.6, 0.4, 0.9,  0.40, 0.75, 0.90),
    ]


# NHS/CRUK target stage distributions (% I/II, % III, % IV)
# Sources: NLCA 2022-2023, NHS Digital 2015-2019, CRUK, Nuffield Trust
NHS_TARGETS = {
    'Anus':       (55, 25, 20),
    'Bladder':    (52, 20, 28),
    'Colorectal': (44, 30, 26),
    'Esophagus':  (15, 25, 60),
    'Head_Neck':  (35, 25, 40),
    'Liver_Bile': (15, 20, 65),
    'Lung':       (34, 22, 44),
    'Lymphoma':   (40, 22, 38),
    'Myeloma':    (40, 28, 32),   # approximate — staging differs for haem
    'Ovary':      (30, 46, 24),
    'Pancreas':   (12, 25, 63),
    'Stomach':    (20, 25, 55),
}


# ============================================================================
# SIMULATION ENGINE (identical to v2)
# ============================================================================

def sample_lognormal(rng, median, sigma):
    return np.exp(np.log(max(median, 1e-12)) + sigma * rng.standard_normal())

def sample_delay_years(rng, median_days, sigma):
    return np.exp(np.log(max(median_days / 365.25, 1e-12)) + sigma * rng.standard_normal())

def apply_spillover(cfg, med_days, t_init):
    if cfg.enable_spillover and 0 <= t_init < cfg.spillover_t_end:
        return med_days * cfg.spillover_factor
    return med_days

def stage_at_time(t0, dE, dIII, dIV, t):
    if t < t0: return 0
    if t < t0 + dE: return 2
    if t < t0 + dE + dIII: return 3
    return 4

def stage_bin(st):
    if st <= 2: return 0
    if st == 3: return 1
    return 2

def sens_by_stage(cancer, st):
    if st <= 2: return cancer.sensE
    if st == 3: return cancer.sensIII
    return cancer.sensIV

def sample_referral_time(rng, t0, dE, dIII, dIV, kE, kIII, kIV):
    tE_end = t0 + dE
    tIII_end = tE_end + dIII
    tIV_end = tIII_end + dIV
    w = rng.exponential(1.0 / max(kE, 1e-12))
    if t0 + w < tE_end: return t0 + w
    w = rng.exponential(1.0 / max(kIII, 1e-12))
    if tE_end + w < tIII_end: return tE_end + w
    w = rng.exponential(1.0 / max(kIV, 1e-12))
    if tIII_end + w < tIV_end: return tIII_end + w
    return np.inf


def simulate_arm(cfg, cancers, arm, rng):
    nC = len(cancers)
    weights = np.array([c.weight for c in cancers])
    weights = weights / weights.sum()
    N_total = round(cfg.N_screened * cfg.incidence_frac)
    N_by_cancer = np.maximum(1, np.round(N_total * weights).astype(int))

    counts_stage = np.zeros((nC, 3), dtype=int)
    iv_by_interval = np.zeros(3, dtype=int)
    slip_soc = np.zeros((3, 3), dtype=int)
    slip_scr = np.zeros((3, 3), dtype=int)
    slip_by_cancer_soc = np.zeros((nC, 3, 3), dtype=int)
    slip_by_cancer_scr = np.zeros((nC, 3, 3), dtype=int)

    for ci in range(nC):
        c = cancers[ci]
        for _ in range(N_by_cancer[ci]):
            t0 = -cfg.T_pre + (cfg.T_pre + cfg.T_follow) * rng.random()
            dE = sample_lognormal(rng, c.medE, cfg.sojourn_sigma)
            dIII = sample_lognormal(rng, c.medIII, cfg.sojourn_sigma)
            dIV = sample_lognormal(rng, c.medIV, cfg.sojourn_sigma)
            t_death = t0 + dE + dIII + sample_lognormal(rng, c.medSurvIV, 0.45)

            t_ref = sample_referral_time(rng, t0, dE, dIII, dIV, c.kE, c.kIII, c.kIV)
            if np.isinf(t_ref):
                t_ref = t0 + dE + dIII + dIV

            soc_med = apply_spillover(cfg, cfg.soc_delay_median_days, t_ref)
            t_dx_soc = t_ref + sample_delay_years(rng, soc_med, cfg.delay_sigma)

            t_dx_scr_best = np.inf
            t_init_scr_best = np.nan
            st_init_scr_best = 0

            if arm == "intervention":
                for r, ts in enumerate(cfg.screen_times):
                    if ts < t0: continue
                    if ts >= t_dx_soc: break
                    st = stage_at_time(t0, dE, dIII, dIV, ts)
                    if rng.random() <= sens_by_stage(c, st):
                        med = cfg.screen_delay_median_days[r]
                        if not cfg.enable_learning:
                            med = cfg.screen_delay_median_days[0]
                        med = apply_spillover(cfg, med, ts)
                        t_dx_scr = ts + sample_delay_years(rng, med, cfg.delay_sigma)
                        if t_dx_scr < t_dx_scr_best:
                            t_dx_scr_best = t_dx_scr
                            t_init_scr_best = ts
                            st_init_scr_best = st

            route = "soc"; t_init = t_ref; t_dx = t_dx_soc
            st_init = stage_at_time(t0, dE, dIII, dIV, t_init)
            if arm == "intervention" and t_dx_scr_best < t_dx_soc:
                route = "screen"; t_init = t_init_scr_best
                t_dx = t_dx_scr_best; st_init = st_init_scr_best

            if t_death < t_dx: continue
            if t_dx < 0 or t_dx >= cfg.T_follow: continue

            st_dx = stage_at_time(t0, dE, dIII, dIV, t_dx)

            if cfg.enable_upstaging:
                if route == "screen" and rng.random() < cfg.upstage_prob_screen:
                    st_dx = min(4, st_dx + 1)
                elif route == "soc" and rng.random() < cfg.upstage_prob_soc:
                    st_dx = min(4, st_dx + 1)

            idx_init = stage_bin(st_init)
            idx_dx = stage_bin(st_dx)
            counts_stage[ci, idx_dx] += 1

            if st_dx == 4:
                k = min(2, max(0, int(np.floor(t_dx))))
                iv_by_interval[k] += 1

            if route == "soc":
                slip_soc[idx_init, idx_dx] += 1
                slip_by_cancer_soc[ci, idx_init, idx_dx] += 1
            else:
                slip_scr[idx_init, idx_dx] += 1
                slip_by_cancer_scr[ci, idx_init, idx_dx] += 1

    return {
        'counts_stage': counts_stage, 'iv_by_interval': iv_by_interval,
        'slip_soc': slip_soc, 'slip_scr': slip_scr,
        'slip_by_cancer_soc': slip_by_cancer_soc,
        'slip_by_cancer_scr': slip_by_cancer_scr,
    }


# ============================================================================
# CALIBRATION ENGINE
# ============================================================================

def simulate_single_cancer_control(cfg, cancer, n_cases, rng):
    """Run control arm for a single cancer type, return stage counts."""
    counts = np.zeros(3, dtype=int)
    for _ in range(n_cases):
        t0 = -cfg.T_pre + (cfg.T_pre + cfg.T_follow) * rng.random()
        dE = sample_lognormal(rng, cancer.medE, cfg.sojourn_sigma)
        dIII = sample_lognormal(rng, cancer.medIII, cfg.sojourn_sigma)
        dIV = sample_lognormal(rng, cancer.medIV, cfg.sojourn_sigma)
        t_death = t0 + dE + dIII + sample_lognormal(rng, cancer.medSurvIV, 0.45)

        t_ref = sample_referral_time(rng, t0, dE, dIII, dIV,
                                     cancer.kE, cancer.kIII, cancer.kIV)
        if np.isinf(t_ref):
            t_ref = t0 + dE + dIII + dIV

        soc_med = apply_spillover(cfg, cfg.soc_delay_median_days, t_ref)
        t_dx = t_ref + sample_delay_years(rng, soc_med, cfg.delay_sigma)

        if t_death < t_dx: continue
        if t_dx < 0 or t_dx >= cfg.T_follow: continue

        st_dx = stage_at_time(t0, dE, dIII, dIV, t_dx)
        counts[stage_bin(st_dx)] += 1

    return counts


def evaluate_hazards(cfg, cancer, kE, kIII, kIV, target, n_seeds=30, n_cases=300):
    """Evaluate how well a set of hazards matches the target distribution."""
    c = copy.copy(cancer)
    c.kE = kE; c.kIII = kIII; c.kIV = kIV

    total_counts = np.zeros(3, dtype=float)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed * 1000 + hash(cancer.name) % 10000)
        counts = simulate_single_cancer_control(cfg, c, n_cases, rng)
        total_counts += counts

    total = total_counts.sum()
    if total < 10:
        return 1e6

    sim_pct = 100.0 * total_counts / total
    target_arr = np.array(target, dtype=float)

    # Weighted squared error — weight Stage I/II match more heavily
    # since that's where the calibration gap is largest
    err = np.sum((sim_pct - target_arr) ** 2)
    return err


def calibrate_cancer(cfg, cancer, target):
    """Grid search over (kE, kIII, kIV) to match target stage distribution."""
    # Search space: kE from 0.1 to 2.0, kIII from 0.5 to 4.0, kIV from 1.0 to 6.0
    # But maintain kE < kIII < kIV

    best_err = 1e9
    best_k = (cancer.kE, cancer.kIII, cancer.kIV)

    kE_range = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00]
    kIII_range = [0.50, 0.80, 1.00, 1.50, 2.00, 2.50, 3.00, 4.00]
    kIV_range = [1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00]

    for kE in kE_range:
        for kIII in kIII_range:
            if kIII <= kE:
                continue
            for kIV in kIV_range:
                if kIV <= kIII:
                    continue
                err = evaluate_hazards(cfg, cancer, kE, kIII, kIV, target,
                                      n_seeds=15, n_cases=200)
                if err < best_err:
                    best_err = err
                    best_k = (kE, kIII, kIV)

    # Refine around best with finer grid
    kE_b, kIII_b, kIV_b = best_k
    kE_fine = [max(0.05, kE_b - 0.15), kE_b - 0.05, kE_b, kE_b + 0.05, kE_b + 0.15]
    kIII_fine = [max(0.2, kIII_b - 0.3), kIII_b - 0.1, kIII_b, kIII_b + 0.1, kIII_b + 0.3]
    kIV_fine = [max(0.5, kIV_b - 0.5), kIV_b - 0.2, kIV_b, kIV_b + 0.2, kIV_b + 0.5]

    for kE in kE_fine:
        for kIII in kIII_fine:
            if kIII <= kE: continue
            for kIV in kIV_fine:
                if kIV <= kIII: continue
                err = evaluate_hazards(cfg, cancer, kE, kIII, kIV, target,
                                      n_seeds=25, n_cases=250)
                if err < best_err:
                    best_err = err
                    best_k = (kE, kIII, kIV)

    return best_k, best_err


# ============================================================================
# SUMMARY + OUTPUT (from v2)
# ============================================================================

def summarize(ctl, intv, cancers):
    s = {}
    s['ctl_stage'] = ctl['counts_stage'].sum(axis=0)
    s['int_stage'] = intv['counts_stage'].sum(axis=0)
    s['ctl_total'] = s['ctl_stage'].sum()
    s['int_total'] = s['int_stage'].sum()
    s['ctl_IIIIV'] = s['ctl_stage'][1] + s['ctl_stage'][2]
    s['int_IIIIV'] = s['int_stage'][1] + s['int_stage'][2]
    s['RR_IIIIV'] = s['int_IIIIV'] / max(1, s['ctl_IIIIV'])
    s['ctl_IV'] = s['ctl_stage'][2]
    s['int_IV'] = s['int_stage'][2]
    s['RR_IV'] = s['int_IV'] / max(1, s['ctl_IV'])
    s['ctl_iv_interval'] = ctl['iv_by_interval']
    s['int_iv_interval'] = intv['iv_by_interval']
    s['slip_soc'] = ctl['slip_soc']
    s['slip_scr'] = intv['slip_scr']
    s['ctl_bycancer'] = ctl['counts_stage']
    s['int_bycancer'] = intv['counts_stage']
    return s


def slip_stats(M):
    initE = M[0, :].sum()
    slipE = M[0, 1] + M[0, 2]
    pctE = 100 * slipE / max(1, initE)
    initIII = M[1, :].sum()
    slipIII = M[1, 2]
    pctIII = 100 * slipIII / max(1, initIII)
    return initE, slipE, pctE, initIII, slipIII, pctIII


def run_multi_seed(cfg, cancers):
    results = []
    for seed in range(cfg.n_seeds):
        rng = np.random.default_rng(seed)
        ctl = simulate_arm(cfg, cancers, "control", rng)
        rng2 = np.random.default_rng(seed + 100_000)
        intv = simulate_arm(cfg, cancers, "intervention", rng2)
        s = summarize(ctl, intv, cancers)
        results.append(s)
        if (seed + 1) % 50 == 0:
            print(f"    {seed+1}/{cfg.n_seeds} seeds", flush=True)
    return results


def aggregate_results(results):
    keys = ['RR_IIIIV', 'RR_IV', 'ctl_IIIIV', 'int_IIIIV', 'ctl_IV', 'int_IV',
            'ctl_total', 'int_total']
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in results])
        agg[k] = {'mean': np.mean(vals), 'lo': np.percentile(vals, 2.5),
                  'hi': np.percentile(vals, 97.5)}

    for stage_label, idx in [('E', 0), ('III', 1), ('IV', 2)]:
        for arm in ['ctl', 'int']:
            vals = np.array([r[f'{arm}_stage'][idx] for r in results])
            agg[f'{arm}_{stage_label}'] = {
                'mean': np.mean(vals), 'lo': np.percentile(vals, 2.5),
                'hi': np.percentile(vals, 97.5)}

    for label, key in [('soc', 'slip_soc'), ('scr', 'slip_scr')]:
        eSlip, iiiSlip = [], []
        for r in results:
            _, sE, pE, _, sIII, pIII = slip_stats(r[key])
            eSlip.append(pE); iiiSlip.append(pIII)
        agg[f'slip_{label}_E_pct'] = {
            'mean': np.mean(eSlip), 'lo': np.percentile(eSlip, 2.5),
            'hi': np.percentile(eSlip, 97.5)}
        agg[f'slip_{label}_III_pct'] = {
            'mean': np.mean(iiiSlip), 'lo': np.percentile(iiiSlip, 2.5),
            'hi': np.percentile(iiiSlip, 97.5)}

    return agg


def delay_sweep(cfg, cancers, delays=None, n_seeds_sweep=50):
    if delays is None:
        delays = [65, 75, 85, 92, 100, 110, 120]
    sweep_results = []
    for d in delays:
        print(f"  Sweep: delay = {d} days ...", flush=True)
        cfg_d = Config(
            N_screened=cfg.N_screened, incidence_frac=cfg.incidence_frac,
            T_follow=cfg.T_follow, T_pre=cfg.T_pre,
            screen_times=cfg.screen_times,
            delay_sigma=cfg.delay_sigma, sojourn_sigma=cfg.sojourn_sigma,
            soc_delay_median_days=d,
            screen_delay_median_days=cfg.screen_delay_median_days,
            enable_spillover=cfg.enable_spillover,
            spillover_factor=cfg.spillover_factor,
            spillover_t_end=cfg.spillover_t_end,
            enable_learning=cfg.enable_learning,
            enable_upstaging=cfg.enable_upstaging,
            n_seeds=n_seeds_sweep)
        res_list = run_multi_seed(cfg_d, cancers)
        agg = aggregate_results(res_list)
        sweep_results.append({
            'delay': d,
            'RR_IIIIV_mean': agg['RR_IIIIV']['mean'],
            'RR_IIIIV_lo': agg['RR_IIIIV']['lo'],
            'RR_IIIIV_hi': agg['RR_IIIIV']['hi'],
            'RR_IV_mean': agg['RR_IV']['mean'],
            'RR_IV_lo': agg['RR_IV']['lo'],
            'RR_IV_hi': agg['RR_IV']['hi'],
        })
    return sweep_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()
    cancers = cancer_table_12()
    outdir = os.path.join(os.getcwd(), 'out_v3')
    os.makedirs(outdir, exist_ok=True)

    # ---- STEP 1: CALIBRATE ----
    print("=" * 72)
    print("STEP 1: CALIBRATING referral hazards to NHS/CRUK stage distributions")
    print("=" * 72)

    for ci, c in enumerate(cancers):
        target = NHS_TARGETS.get(c.name)
        if target is None:
            print(f"  {c.name}: no target, keeping defaults")
            continue

        print(f"  Calibrating {c.name} (target: I/II={target[0]}%, III={target[1]}%, IV={target[2]}%) ...", flush=True)
        best_k, best_err = calibrate_cancer(cfg, c, target)
        c.kE, c.kIII, c.kIV = best_k
        cancers[ci] = c
        print(f"    -> kE={best_k[0]:.2f}, kIII={best_k[1]:.2f}, kIV={best_k[2]:.2f}  (err={best_err:.1f})")

    # ---- Verify calibration ----
    print(f"\n{'Cancer':<14s}  {'kE':>5s} {'kIII':>5s} {'kIV':>5s}  |  {'Sim I/II':>8s} {'Sim III':>8s} {'Sim IV':>8s}  |  {'NHS I/II':>8s} {'NHS III':>8s} {'NHS IV':>8s}  |  {'Gap':>6s}")
    print("-" * 105)

    for ci, c in enumerate(cancers):
        total_counts = np.zeros(3, dtype=float)
        for seed in range(50):
            rng = np.random.default_rng(seed * 1000 + ci)
            counts = simulate_single_cancer_control(cfg, c, 300, rng)
            total_counts += counts
        total = total_counts.sum()
        sim_pct = 100.0 * total_counts / max(1, total)
        target = NHS_TARGETS.get(c.name, (0, 0, 0))
        gap = np.sqrt(np.mean((sim_pct - np.array(target)) ** 2))
        print(f"  {c.name:<12s}  {c.kE:>5.2f} {c.kIII:>5.2f} {c.kIV:>5.2f}  |  "
              f"{sim_pct[0]:>8.1f} {sim_pct[1]:>8.1f} {sim_pct[2]:>8.1f}  |  "
              f"{target[0]:>8.0f} {target[1]:>8.0f} {target[2]:>8.0f}  |  {gap:>6.1f}")

    # Save calibrated hazards
    with open(os.path.join(outdir, 'calibrated_hazards.txt'), 'w') as f:
        f.write("Cancer,kE,kIII,kIV\n")
        for c in cancers:
            f.write(f"{c.name},{c.kE:.3f},{c.kIII:.3f},{c.kIV:.3f}\n")

    # ---- STEP 2: FULL SIMULATION ----
    print("\n" + "=" * 72)
    print("STEP 2: FULL SIMULATION with calibrated hazards (200 seeds)")
    print("=" * 72)

    results = run_multi_seed(cfg, cancers)
    agg = aggregate_results(results)

    # ---- STEP 3: DELAY SWEEP ----
    print("\n" + "=" * 72)
    print("STEP 3: DELAY SENSITIVITY SWEEP (calibrated)")
    print("=" * 72)

    sweep = delay_sweep(cfg, cancers, n_seeds_sweep=50)

    # ---- PRINT REPORT ----
    def ci(d, fmt=".3f"):
        return f"{d['mean']:{fmt}} [{d['lo']:{fmt}}, {d['hi']:{fmt}}]"

    print("\n" + "=" * 72)
    print("CALIBRATED SIMULATION — RESULTS SUMMARY")
    print(f"  N_screened = {cfg.N_screened:,}  |  SoC delay = {cfg.soc_delay_median_days:.0f} days")
    print(f"  Seeds = {cfg.n_seeds}  |  Screen delays = {cfg.screen_delay_median_days}")
    print("=" * 72)

    print(f"\n--- Overall Endpoints (mean [95% CI]) ---")
    for stage, lab in [('E', 'Stage I/II'), ('III', 'Stage III'), ('IV', 'Stage IV')]:
        print(f"  {lab:14s}  Ctl: {ci(agg[f'ctl_{stage}'], '.1f'):>24s}   "
              f"Int: {ci(agg[f'int_{stage}'], '.1f'):>24s}")
    print(f"  RR(III+IV) = {ci(agg['RR_IIIIV'])}")
    print(f"  RR(IV)     = {ci(agg['RR_IV'])}")

    print(f"\n--- Slip Rates (%) ---")
    print(f"  SoC  E→III/IV: {ci(agg['slip_soc_E_pct'], '.1f')}   "
          f"III→IV: {ci(agg['slip_soc_III_pct'], '.1f')}")
    print(f"  Scr  E→III/IV: {ci(agg['slip_scr_E_pct'], '.1f')}   "
          f"III→IV: {ci(agg['slip_scr_III_pct'], '.1f')}")

    print(f"\n--- Delay Sensitivity Sweep ---")
    print(f"  {'Delay':>7s}  {'RR(III+IV)':>28s}  {'RR(IV)':>28s}")
    for row in sweep:
        mark = " <--" if row['delay'] == cfg.soc_delay_median_days else ""
        rr34 = f"{row['RR_IIIIV_mean']:.3f} [{row['RR_IIIIV_lo']:.3f}, {row['RR_IIIIV_hi']:.3f}]"
        rr4 = f"{row['RR_IV_mean']:.3f} [{row['RR_IV_lo']:.3f}, {row['RR_IV_hi']:.3f}]"
        print(f"  {row['delay']:>5.0f}d  {rr34:>28s}  {rr4:>28s}{mark}")

    # ---- WRITE LaTeX ----
    def f3(x): return f"{x:.3f}"
    def f1(x): return f"{x:.1f}"
    def ci_tex(d, fmt=f3): return f"{fmt(d['mean'])} [{fmt(d['lo'])}, {fmt(d['hi'])}]"

    with open(os.path.join(outdir, 'tab_overall_v3.tex'), 'w') as f:
        f.write("% Auto-generated: calibrated overall endpoints with 95% CIs\n")
        f.write(r"\begin{table}[!ht]\centering" + "\n")
        f.write(r"\caption{Overall stage at diagnosis --- calibrated model (mean [95\% CI], "
                + str(cfg.n_seeds) + r" seeds).}" + "\n")
        f.write(r"\label{tab:overall_v3}" + "\n")
        f.write(r"\begin{tabular}{lll}\toprule" + "\n")
        f.write(r"Metric & Control & Intervention \\ \midrule" + "\n")
        for stage, lab in [('E', 'Stage I/II'), ('III', 'Stage III'), ('IV', 'Stage IV')]:
            f.write(f"{lab} & {ci_tex(agg[f'ctl_{stage}'], f1)} & {ci_tex(agg[f'int_{stage}'], f1)} \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(f"RR(Stage III+IV) & -- & {ci_tex(agg['RR_IIIIV'])} \\\\\n")
        f.write(f"RR(Stage IV) & -- & {ci_tex(agg['RR_IV'])} \\\\\n")
        f.write(r"\bottomrule\end{tabular}\end{table}" + "\n")

    with open(os.path.join(outdir, 'tab_slip_v3.tex'), 'w') as f:
        f.write("% Auto-generated: calibrated slip summary with 95% CIs\n")
        f.write(r"\begin{table}[!ht]\centering" + "\n")
        f.write(r"\caption{Stage slip rates --- calibrated model (mean \% [95\% CI], "
                + str(cfg.n_seeds) + r" seeds).}" + "\n")
        f.write(r"\label{tab:slip_v3}" + "\n")
        f.write(r"\begin{tabular}{lll}\toprule" + "\n")
        f.write(r"Route & E$\to$III/IV (\%) & III$\to$IV (\%) \\ \midrule" + "\n")
        f.write(f"SoC (control) & {ci_tex(agg['slip_soc_E_pct'], f1)} & {ci_tex(agg['slip_soc_III_pct'], f1)} \\\\\n")
        f.write(f"Screen (intervention) & {ci_tex(agg['slip_scr_E_pct'], f1)} & {ci_tex(agg['slip_scr_III_pct'], f1)} \\\\\n")
        f.write(r"\bottomrule\end{tabular}\end{table}" + "\n")

    with open(os.path.join(outdir, 'tab_sweep_v3.tex'), 'w') as f:
        f.write("% Auto-generated: calibrated delay sensitivity sweep\n")
        f.write(r"\begin{table}[!ht]\centering" + "\n")
        f.write(r"\caption{Sensitivity of endpoints to SoC delay --- calibrated model (mean [95\% CI]).}" + "\n")
        f.write(r"\label{tab:sweep_v3}" + "\n")
        f.write(r"\begin{tabular}{rll}\toprule" + "\n")
        f.write(r"SoC Delay (days) & RR(Stage III+IV) & RR(Stage IV) \\ \midrule" + "\n")
        for row in sweep:
            rr34 = f"{row['RR_IIIIV_mean']:.3f} [{row['RR_IIIIV_lo']:.3f}, {row['RR_IIIIV_hi']:.3f}]"
            rr4 = f"{row['RR_IV_mean']:.3f} [{row['RR_IV_lo']:.3f}, {row['RR_IV_hi']:.3f}]"
            marker = r" $\leftarrow$" if row['delay'] == cfg.soc_delay_median_days else ""
            f.write(f"{row['delay']}{marker} & {rr34} & {rr4} \\\\\n")
        f.write(r"\bottomrule\end{tabular}\end{table}" + "\n")

    # Calibrated cancer params table
    with open(os.path.join(outdir, 'tab_cancer_v3.tex'), 'w') as f:
        f.write("% Auto-generated: calibrated cancer parameters\n")
        f.write(r"\begin{table}[!ht]\centering\small" + "\n")
        f.write(r"\caption{12-cancer parameters (calibrated referral hazards).}" + "\n")
        f.write(r"\label{tab:cancer_v3}" + "\n")
        f.write(r"\begin{tabular}{lrrrrrrrr}\toprule" + "\n")
        f.write(r"Cancer & $w_c$ & $\tilde{D}_E$ & $\tilde{D}_{III}$ & "
                r"$k_E$ & $k_{III}$ & $k_{IV}$ & "
                r"sens$_E$ & sens$_{IV}$ \\ \midrule" + "\n")
        wtot = sum(c.weight for c in cancers)
        for c in cancers:
            w = c.weight / wtot
            f.write(f"{c.name.replace('_', ' ')} & {w:.3f} & {c.medE:.2f} & "
                    f"{c.medIII:.2f} & {c.kE:.2f} & {c.kIII:.2f} & {c.kIV:.2f} & "
                    f"{c.sensE:.2f} & {c.sensIV:.2f} \\\\\n")
        f.write(r"\bottomrule\end{tabular}\end{table}" + "\n")

    print(f"\n  LaTeX tables written to {outdir}/")
    print("Done.")


if __name__ == "__main__":
    main()
