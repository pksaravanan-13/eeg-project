# EEG Pipeline — Reference Sources by Module

Use these when asking Gemini/AI chats for help — paste the relevant link(s) or paper title as context so answers stay grounded in the actual method your pipeline uses.

---

## Core (reach for these on almost any question)

| Source | Link | Use for |
|---|---|---|
| MNE-Python documentation | https://mne.tools/stable/documentation.html | API reference, general lookup |
| MNE-Python tutorials gallery | https://mne.tools/stable/auto_tutorials/index.html | Runnable worked examples per topic |
| MNE-Python examples gallery | https://mne.tools/stable/auto_examples/index.html | Shorter, targeted code snippets |
| MNE-Python GitHub Discussions | https://github.com/mne-tools/mne-python/discussions | Real troubleshooting threads |
| MNE-Python GitHub Issues | https://github.com/mne-tools/mne-python/issues | Known bugs / edge cases |

---

## M0 — Environment Setup

| Source | Link |
|---|---|
| MNE installation guide | https://mne.tools/stable/install/index.html |
| uv (package manager) docs | https://docs.astral.sh/uv/ |

---

## M1 — Data Loading

| Source | Link |
|---|---|
| MNE "Importing data" overview (all formats) | https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html |
| MNE `Raw` object API | https://mne.tools/stable/generated/mne.io.Raw.html |
| MNE sample dataset docs | https://mne.tools/stable/documentation/datasets.html#sample |
| BIDS-EEG specification | https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html |

---

## M2 — Filtering

| Source | Link |
|---|---|
| MNE "Background on filtering" tutorial | https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html |
| MNE `filter()` / `raw.filter` API | https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter |
| Widmann, Schröger & Maess (2015) — *Digital filter design for electrophysiological data* | https://doi.org/10.1016/j.jneumeth.2014.08.002 |
| 3Blue1Brown — Fourier Transform (intuition) | https://www.youtube.com/watch?v=spUNpyF58BY |

---

## M3 — Epoching

| Source | Link |
|---|---|
| MNE `Epochs` API | https://mne.tools/stable/generated/mne.Epochs.html |
| MNE "Epoching and averaging" tutorial | https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html |
| MNE "Working with events" tutorial | https://mne.tools/stable/auto_tutorials/intro/20_events_from_raw.html |

---

## M4 — Artifact Rejection

| Source | Link |
|---|---|
| MNE "Rejecting bad data spans" tutorial | https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html |
| MNE "Handling bad channels" tutorial | https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html |
| Autoreject documentation | https://autoreject.github.io/stable/index.html |
| Autoreject paper (Jas et al., 2017) | https://doi.org/10.1016/j.neuroimage.2017.06.030 |

---

## M5 — ICA

| Source | Link |
|---|---|
| MNE "Repairing artifacts with ICA" tutorial | https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html |
| MNE `ICA` API | https://mne.tools/stable/generated/mne.preprocessing.ICA.html |
| ICLabel paper (Pion-Tonachini et al., 2019) | https://doi.org/10.1016/j.neuroimage.2019.05.026 |

---

## M6 — ERP Analysis

| Source | Link |
|---|---|
| MNE "Visualizing Evoked data" tutorial | https://mne.tools/stable/auto_tutorials/evoked/20_visualize_evoked.html |
| MNE `Evoked` API | https://mne.tools/stable/generated/mne.Evoked.html |
| Luck — *An Introduction to the Event-Related Potential Technique* (book, MIT Press) | https://mitpress.mit.edu/9780262525855/an-introduction-to-the-event-related-potential-technique/ |

---

## M7 — Time-Frequency Analysis

| Source | Link |
|---|---|
| MNE "Time-frequency analysis: power and ITC" tutorial | https://mne.tools/stable/auto_tutorials/time-freq/20_erds.html |
| MNE `tfr_morlet` / time-frequency API | https://mne.tools/stable/generated/mne.time_frequency.tfr_morlet.html |
| Cohen — *Analyzing Neural Time Series Data* (book site + MATLAB/Python code) | http://mikexcohen.com/book/ |

---

## M8 — Connectivity / ITC

| Source | Link |
|---|---|
| MNE-Connectivity documentation | https://mne.tools/mne-connectivity/stable/index.html |
| MNE-Connectivity API reference | https://mne.tools/mne-connectivity/stable/api.html |
| MNE-Connectivity examples gallery | https://mne.tools/mne-connectivity/stable/auto_examples/index.html |

---

## M9 — ML Classification

| Source | Link |
|---|---|
| MNE "Decoding (MVPA)" tutorial | https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html |
| scikit-learn documentation | https://scikit-learn.org/stable/ |
| PhysioNet EEGBCI (Motor Imagery) dataset page | https://physionet.org/content/eegmmidb/1.0.0/ |
| MNE `eegbci` dataset loader docs | https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html |
| Braindecode (deep learning on EEG, optional stretch) | https://braindecode.org/stable/index.html |

---

## General Grounding (concept-level, not code-specific)

| Source | Link |
|---|---|
| Cohen — *Analyzing Neural Time Series Data* | http://mikexcohen.com/book/ |
| Luck — *An Introduction to the Event-Related Potential Technique* | https://mitpress.mit.edu/9780262525855/an-introduction-to-the-event-related-potential-technique/ |
| MNE-Python glossary (terminology lookup) | https://mne.tools/stable/glossary.html |

---

## How to use this with Gemini

1. Find your current module's row above.
2. Paste the doc/tutorial link (or paper title) into Gemini along with your actual code/error.
3. For conceptual gaps (not code bugs), pull from "General Grounding" first before diving into API docs.
