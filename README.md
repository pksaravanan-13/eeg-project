# eeg-project

EEG signal processing and analysis pipeline built around [MNE-Python](https://mne.tools/), developed
milestone by milestone as a learning project toward BCI/neurotech engineering work. Each milestone adds
a `src/` module and a companion notebook that exercises it against the MNE built-in sample dataset (with
a documented path to swap in a local recording).

## Milestone status

| Milestone | Module | Notebook | Status |
|---|---|---|---|
| M1 — Data loading | `src/preprocessing/loader.py` | `notebooks/M1_data_loading.ipynb` | Done |
| M2 — Filtering | `src/preprocessing/filter.py` | `notebooks/M2_filtering.ipynb` | Done |
| M3 — Epoching | `src/preprocessing/epoching.py` | `notebooks/M3_epoching.ipynb` | Done |
| M4 — Artifact rejection | `src/preprocessing/artifacts.py` | `notebooks/M4_artifact_rejection.ipynb` | Done |
| M5 — ICA | — | — | Planned |
| M6 — ERP analysis | `src/analysis/features.py` (partial) | — | Planned |
| M7 — Time-frequency | `src/analysis/features.py` (partial) | — | Planned |
| M8 — Connectivity / ITC | — | — | Planned |
| M9 — ML classification | — | — | Planned |

See `SOURCES.md` for the reference material behind each milestone.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
```

Dependencies are declared in `pyproject.toml` (single source of truth); `requirements.txt` just points
back to it.

## Running the pipeline

`pipeline.py` chains preprocessing (load → filter → epoch → artifact-reject) → analysis (band power, ERP,
time-frequency) → visualization for one subject, and is resumable: it skips stages whose output is already
up to date with the current config, and writes a JSON provenance sidecar next to each processed file
recording exactly what parameters produced it.

```bash
# Single subject
python pipeline.py --subject sub-01 --file path/to/raw.fif --bad-channels "EEG 053"

# Recompute even if cached output looks up to date
python pipeline.py --subject sub-01 --file path/to/raw.fif --force

# Batch mode: iterate config.yaml's `subjects` list, resolving each subject's file
# under paths.raw_data as "{subject}_raw.fif"
python pipeline.py
```

Pipeline parameters (filter cutoffs, epoch window, frequency bands, paths) live in `config.yaml`.

## Project structure

```
config.yaml            pipeline parameters
pipeline.py             CLI entry point: preprocess -> analyze -> visualize, resumable
src/
  preprocessing/        loader, filter, epoching, artifact rejection
  analysis/              band power, ERP, time-frequency (ahead of current milestones)
  visualization/         raw/ERP/topomap/PSD plotting
notebooks/               one exploratory notebook per milestone
tests/                    pytest suite, mirrors src/
data/, results/           gitignored; local raw/processed data and generated figures
```

## Testing

```bash
pytest              # fast suite (synthetic EEG data, no downloads)
pytest -m slow       # add the end-to-end test against the real MNE sample dataset
```
