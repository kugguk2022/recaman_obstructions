# Racaman Obstruction

This repo is organized around three top-level folders:

- `scripts/`: analysis and plotting scripts.
- `outputs/`: generated JSON results and rendered figures.
- `supporting_docs/`: notes, manuscripts, and longer reference documents.

The main input dataset remains at the repo root as `obstructions.txt`.

## 3D Snapshot

Current phase-space render from `scripts/recaman_phase_space_3d.py`:

![Recaman 3D arc-lift render](outputs/recaman_phase_arc_readme.png)

## Current Results

Primary result artifacts:

- [`outputs/version_c_obstructions_results.json`](outputs/version_c_obstructions_results.json)
- [`outputs/recaman_wheel_results.json`](outputs/recaman_wheel_results.json)
- [`outputs/best_obstructions_random_20260512_172100.json`](outputs/best_obstructions_random_20260512_172100.json)

Version C obstruction modeling on `obstructions.txt` using forward CV with `purge_contexts=1`:

- Dataset `A`: mean AUC `0.9961`
- Dataset `B`: mean AUC `0.9964`
- Dataset `C`: mean AUC `0.9944`
- Dataset `D`: mean AUC `0.7586`
- Event summary: `3102` events total, with `2535` singletons and `567` ranges

Wheel / phase-slip validation from [`outputs/recaman_wheel_results.json`](outputs/recaman_wheel_results.json):

- `Theta_3` wheel is falsified: `q_210 = 0.500007`, `q_321 = 0.499996`
- Bit-history separation is dominant: `|Δq| = 0.997832`
- Measured phase-slip rate: `0.001084` (`1.0839 x 10^-3`)
- Logistic 4-feature closure accuracy: `0.98575`

## Layout

```text
.
|-- README.md
|-- obstructions.txt
|-- scripts/
|-- outputs/
`-- supporting_docs/
```

## Main Scripts

- `scripts/321_210_randmat.py`: random matrix search over obstruction-vs-control features.
- `scripts/321_210_version_c.py`: version C obstruction modeling with forward CV support.
- `scripts/recaman_wheel_validator.py`: long-run wheel and phase-slip validation.
- `scripts/recaman_wheel_honest.py`: honest wheel null comparison.
- `scripts/recaman_modm_scan.py`: modular state scan for obstruction separation.
- `scripts/recaman_heldout.py`: held-out checks for candidate predictors.
- `scripts/recaman_phase_space_3d.py`: 3D phase-space plots for the Recaman sequence.
- `scripts/recaman_seq_distribution.py`: large-sample distribution histogram.

## Typical Usage

Run from the repo root.

```powershell
python .\scripts\321_210_version_c.py --input-file .\obstructions.txt --datasets ABCD
python .\scripts\321_210_randmat.py --save-best-file .\outputs\best_obstructions_random.json
python .\scripts\recaman_wheel_validator.py
python .\scripts\recaman_phase_space_3d.py --steps 2800 --mode delay --tau 2 --save .\outputs\recaman_phase_delay.png
python .\scripts\recaman_phase_space_3d.py --steps 2800 --mode arc-lift --twist 1.5 --save .\outputs\recaman_phase_arc.png
```

## Supporting Docs

- [`supporting_docs/twistor_splitor_recaman_dossier.pdf`](supporting_docs/twistor_splitor_recaman_dossier.pdf)
- [`supporting_docs/recaman_final_math.md`](supporting_docs/recaman_final_math.md)
- [`supporting_docs/recaman_final_math.pdf`](supporting_docs/recaman_final_math.pdf)
- [`supporting_docs/Recaman_Wheel_Validation.docx`](supporting_docs/Recaman_Wheel_Validation.docx)

## Notes

- `scripts/recaman_wheel_validator.py` now writes its default JSON output into `outputs/`.
- `scripts/recaman_seq_distribution.py` now writes its histogram into `outputs/`.
- Existing historical outputs were moved into `outputs/` rather than deleted.
