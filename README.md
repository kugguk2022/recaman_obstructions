# Racaman Obstruction

This repo is organized around three top-level folders:

- `scripts/`: analysis and plotting scripts.
- `outputs/`: generated JSON results and rendered figures.
- `supporting_docs/`: notes, manuscripts, and longer reference documents.

The main input dataset remains at the repo root as `obstructions.txt`.

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

## Notes

- `scripts/recaman_wheel_validator.py` now writes its default JSON output into `outputs/`.
- `scripts/recaman_seq_distribution.py` now writes its histogram into `outputs/`.
- Existing historical outputs were moved into `outputs/` rather than deleted.
