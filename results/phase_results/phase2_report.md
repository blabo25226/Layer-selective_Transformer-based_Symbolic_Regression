# Phase 2: baseline evaluation

- Suite: `results/synthetic/phase1_v1`
- Split: **test** (9 problems)
- Results: `results/phase_results/phase2/baseline_results.jsonl`
- Summary JSON: `results/phase_results/phase2/baseline_summary.json`

## Methods (v1)

| method | meaning |
|--------|---------|
| `nesymres_beam2` | pretrained NeSymReS, beam_size=2 |
| `nesymres_beam5` | pretrained NeSymReS, beam_size=5 |
| `pysr` | PySR (20 iterations) |
| TPSR | **deferred** (follow-up; heavier adapter) |

## Overall summary (test)

| method | n | valid | median NMSE (ID) | median NMSE (OOD) | median R² (ID) | mean var-F1 | mean time (s) |
|--------|---|-------|------------------|-------------------|----------------|-------------|---------------|
| `nesymres_beam2` | 9 | 1.00 | 0.376 | 0.901 | 0.602 | 1.00 | 20.0 |
| `nesymres_beam5` | 9 | 1.00 | 0.120 | 0.187 | 0.806 | 1.00 | 55.9 |
| `pysr` | 9 | 1.00 | **0.00072** | 0.327 | **0.999** | 0.93 | 15.3 |

## Per-family median NMSE (in-domain)

| family | nesymres_beam2 | nesymres_beam5 | pysr |
|--------|----------------|----------------|------|
| activation | 0.122 | 0.076 | 0.00063 |
| repression | 16.16 | 0.093 | 0.00042 |
| toggle | 0.100 | 0.085 | 0.018 |
| repressilator | 1.092 | 0.282 | 0.0074 |

## Findings

1. **PySR dominates ID fit** on this suite (as expected for small symbolic RHS problems).
2. **Larger beam helps NeSymReS** (beam5 ≪ beam2 on median NMSE), but still far from PySR in-domain.
3. **OOD**: beam5 NeSymReS can beat PySR on median OOD NMSE here (0.19 vs 0.33) — keep as a signal, not a firm claim yet (n=9).
4. Plan note holds: NeSymReS underperforming vs PySR does **not** stop the project; Phase 3+ tests whether layer-selective fine-tuning closes the gap.

## How to reproduce

```powershell
conda activate ltsr-phase0
python scripts/phase2_run_baselines.py --split test --pysr-iters 20
```

## Phase 2 status

- Metrics module: done (`src/evaluation/equation_metrics.py`)
- NeSymReS adapter: done (`src/models/nesymres_adapter.py`)
- PySR + NeSymReS baselines on test: done
- Full symbolic equivalence / sign recovery / TPSR: later
