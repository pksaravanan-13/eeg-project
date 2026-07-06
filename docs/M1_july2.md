# M1 Session — July 2 (3 hrs)
**Goal:** Create `src/preprocessing/loader.py`, clean up `filter.py`, start M1 notebook. Three commits by end of session.

---

## PART 1 — Open VS Code and Set Up (10 min)

### 1.1 Open VS Code
- Launch VS Code from Start Menu or taskbar.

### 1.2 Open the project folder
- `File → Open Folder`
- Navigate to: `C:\Users\ke725\Documents\eeg-project`
- Click **Select Folder**
- You should see the file tree on the left:
  ```
  eeg-project/
  ├── src/
  ├── notebooks/
  ├── data/
  ├── pipeline.py
  ├── requirements.txt
  └── ...
  ```

### 1.3 Open the integrated terminal
- Press `` Ctrl + ` `` (backtick key, top-left of keyboard)
- A terminal panel opens at the bottom of VS Code
- It should already show `C:\Users\ke725\Documents\eeg-project>`

### 1.4 Activate the virtual environment
Type this exactly (single dot, then venv):
```powershell
.venv\Scripts\activate
```
**Expected output:** The prompt changes to:
```
(.venv) PS C:\Users\ke725\Documents\eeg-project>
```
If you see `(.venv)` at the start — you're good. If not, check that you're in the right folder first with:
```powershell
pwd
```
It must show `C:\Users\ke725\Documents\eeg-project`.

### 1.5 Verify MNE is available
```powershell
python -c "import mne; print(mne.__version__)"
```
**Expected output:** `1.12.1`

---

## PART 2 — Git Branch (5 min)

### 2.1 Check current status
```powershell
git status
```
**Expected output:**
```
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```
If you see modified files listed, that's fine — just means existing changes from M0 setup. Don't worry about them now.

### 2.2 Create and switch to the M1 branch
```powershell
git checkout -b feature/M1-data-loading
```
- `git checkout` = switch branches
- `-b` = create a new branch at the same time (without `-b` it would just switch to an existing branch)
- `feature/M1-data-loading` = the branch name — the `feature/` prefix is convention to show this branch adds a new feature

**Expected output:**
```
Switched to a new branch 'feature/M1-data-loading'
```

### 2.3 Confirm you're on the right branch
```powershell
git branch
```
**Expected output:**
```
* feature/M1-data-loading
  main
```
The `*` must be next to `feature/M1-data-loading`.

---

## PART 3 — Read the Existing filter.py First (5 min)

Before writing anything, open and read what already exists.

### 3.1 Open the file
In the VS Code left panel, click:
`src → preprocessing → filter.py`

You'll see:
```python
import mne
from pathlib import Path

def load_raw(filepath: str) -> mne.io.Raw:
    return mne.io.read_raw(filepath, preload=True)

def apply_filters(raw, l_freq, h_freq, notch_freq):
    raw.notch_filter(notch_freq)
    raw.filter(l_freq, h_freq)
    return raw

def make_epochs(raw, tmin, tmax):
    events, event_id = mne.events_from_annotations(raw)
    return mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, preload=True)

def save_processed(epochs, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    epochs.save(out_path, overwrite=True)
```

**Note:** `load_raw` is here right now. You are going to move it into its own file (`loader.py`) and upgrade it. Then remove it from `filter.py` and replace it with an import. This separation matters because loading data and filtering data are two different responsibilities — keeping them in separate files makes the codebase easier to navigate and test.

---

## PART 4 — Create loader.py (45 min)

### 4.1 Create the new file
In the VS Code left panel:
- Right-click on `src/preprocessing/`
- Click **New File**
- Name it: `loader.py`

### 4.2 Type this code into loader.py line by line

Type this yourself — do not paste. Typing it forces you to read and think about every line.

```python
# Line 1: import mne
# mne is the core library for EEG/MEG processing in Python.
# It provides the Raw object (which holds your EEG signal),
# the Epochs object (time-locked segments), and all the
# processing functions we use throughout M1–M9.
import mne

# Line 2: from pathlib import Path
# pathlib is a Python built-in module for handling file paths.
# Path objects understand Windows backslashes AND Mac/Linux forward
# slashes automatically — so our code runs on any OS without changes.
# It also gives us .suffix (to check the file extension) and
# .parent (to get the folder a file lives in).
from pathlib import Path


# Lines 5–14: def load_raw(...)
# This is the function that opens an EEG file from disk and
# loads it into memory as an MNE Raw object.
# Every single module downstream (M2 filtering, M3 epoching,
# M4 artifacts, M5 ICA...) starts by calling this function.
# It is the entry point of the entire pipeline.
#
# file_path: str  — the full path to the EEG file on disk,
#                   passed in as a plain string (e.g. "C:/data/sub01.fif")
# preload: bool = True — whether to load the full signal into RAM now.
#                        True is required before filtering. Default is True
#                        so callers don't need to think about it.
# -> mne.io.Raw   — the return type annotation. Tells anyone reading
#                   this code exactly what comes back: an MNE Raw object.
def load_raw(file_path: str, preload: bool = True) -> mne.io.Raw:

    # Convert the string path into a Path object.
    # This gives us access to .suffix, .stem, .parent, etc.
    # Without this conversion, we'd have to manually parse the
    # string to find the file extension.
    path = Path(file_path)

    # Check the file extension to decide which MNE reader to use.
    # MNE has separate reader functions for each format because
    # .fif and .edf store data differently on disk.
    if path.suffix == ".fif":
        # .fif is MNE's native format. The MNE sample dataset
        # (which we use for M1–M8) is in this format.
        # read_raw_fif is the fastest and most full-featured reader.
        raw = mne.io.read_raw_fif(path, preload=preload)

    elif path.suffix == ".edf":
        # .edf (European Data Format) is the standard clinical EEG format.
        # Most hospital EEG systems export .edf files.
        # You will almost certainly encounter this format in real research.
        raw = mne.io.read_raw_edf(path, preload=preload)

    else:
        # If neither format matches, raise a clear error immediately.
        # This is better than letting Python crash later with a confusing
        # message — we tell the user exactly what went wrong and why.
        # f-string shows the actual extension they passed in.
        raise ValueError(f"Unsupported format: {path.suffix}")

    # Return the Raw object so the caller (notebook or pipeline.py)
    # can use it for filtering, epoching, etc.
    return raw


# Lines 17–21: def inspect_raw(...)
# A diagnostic function — not part of the analysis, just for you
# to verify that the file loaded correctly before proceeding.
# Always call this right after load_raw to catch problems early
# (wrong file, wrong channels, unexpected sampling rate).
#
# raw: mne.io.Raw — takes the Raw object returned by load_raw
# -> None          — returns nothing, just prints to screen
def inspect_raw(raw: mne.io.Raw) -> None:

    # raw.ch_names is a Python list of every channel name
    # (e.g. ['EEG 001', 'EEG 002', ..., 'MEG 0111', ...])
    # len() gives us the total count.
    # For the MNE sample dataset, this will be 376 (EEG + MEG + misc channels).
    print(f"Channels    : {len(raw.ch_names)}")

    # raw.info is a dictionary-like object containing all metadata
    # about the recording. 'sfreq' is the sampling frequency in Hz —
    # how many data points are recorded per second per channel.
    # Higher sfreq = more temporal resolution but bigger files.
    # The MNE sample dataset is ~600 Hz. Typical EEG-only datasets are 250–1000 Hz.
    print(f"Sampling rate: {raw.info['sfreq']} Hz")

    # raw.times is a numpy array of all time points in seconds.
    # The last element [-1] is the total duration of the recording.
    # :.1f formats it to 1 decimal place (e.g. 277.0 seconds).
    print(f"Duration    : {raw.times[-1]:.1f} seconds")

    # get_channel_types() returns a list with one entry per channel
    # showing its type: 'eeg', 'grad', 'mag', 'stim', 'eog', etc.
    # Wrapping in set() removes duplicates so we see each type once.
    # This tells us what kinds of sensors are in the file —
    # important because EEG and MEG channels are processed differently.
    print(f"Channel types: {set(raw.get_channel_types())}")
```

### 4.3 Save the file
Press `Ctrl + S`

### 4.4 Quick sanity test — run it in terminal
```powershell
python -c "from src.preprocessing.loader import load_raw, inspect_raw; print('loader.py imported successfully')"
```
**Expected output:**
```
loader.py imported successfully
```
If you get an ImportError, check that the file is saved inside `src/preprocessing/` and there are no typos in the function names.

---

## PART 5 — Update filter.py (15 min)

Now remove `load_raw` from `filter.py` and replace it with an import from `loader.py`.

### 5.1 Open filter.py again
Click `src → preprocessing → filter.py`

### 5.2 Replace the entire contents with this
Select all with `Ctrl + A`, then type the new version:

```python
# import mne — the MNE library for all EEG processing objects and methods.
import mne

# pathlib.Path — used in save_processed to create directories safely.
from pathlib import Path

# We no longer define load_raw here — it now lives in loader.py.
# Importing it here means pipeline.py can still do:
#   from src.preprocessing.filter import load_raw
# and it will work, because Python re-exports imported names.
from src.preprocessing.loader import load_raw


# apply_filters applies two sequential filters to the raw signal:
# 1. Notch filter — removes power line noise (60 Hz in the US)
# 2. Bandpass filter — keeps only frequencies between l_freq and h_freq
#
# raw: mne.io.Raw   — the loaded, unfiltered EEG signal
# l_freq: float     — low cutoff frequency in Hz (default 1.0 in pipeline.py)
#                     removes slow DC drift below this frequency
# h_freq: float     — high cutoff frequency in Hz (default 40.0 in pipeline.py)
#                     removes muscle noise above this frequency
# notch_freq: float — the frequency to notch out (60.0 Hz for US power lines)
# -> mne.io.Raw     — returns the same Raw object, now filtered in place
def apply_filters(raw: mne.io.Raw, l_freq: float, h_freq: float, notch_freq: float) -> mne.io.Raw:
    # notch_filter removes a narrow band around 60 Hz.
    # Power line interference at exactly 60 Hz contaminates every EEG recording
    # made with plugged-in equipment. This removes it cleanly.
    raw.notch_filter(notch_freq)

    # filter applies the bandpass — keeps 1–40 Hz, removes everything outside.
    # We apply notch first, then bandpass, because the notch target (60 Hz)
    # is outside the bandpass range anyway and this order avoids edge artifacts.
    raw.filter(l_freq, h_freq)

    return raw


# make_epochs cuts the continuous filtered signal into short time windows
# locked to stimulus events (button presses, sounds, visual flashes).
# Each window is one "trial" — the raw data equivalent of a single observation.
#
# raw: mne.io.Raw — filtered continuous signal
# tmin: float     — seconds before each event to include (negative = before)
# tmax: float     — seconds after each event to include
# -> mne.Epochs   — an MNE Epochs object: a 3D array of (trials × channels × timepoints)
def make_epochs(raw: mne.io.Raw, tmin: float, tmax: float) -> mne.Epochs:
    # events_from_annotations reads the stimulus channel embedded in the Raw object
    # and returns: events (array of [sample_index, 0, event_id]) and
    # event_id (dict mapping event name → integer code).
    events, event_id = mne.events_from_annotations(raw)

    # mne.Epochs slices the Raw signal at each event and stacks the windows.
    # preload=True loads all epochs into RAM immediately —
    # required for artifact rejection and ICA in later modules.
    return mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, preload=True)


# save_processed writes the cleaned, epoched data to disk as a .fif file
# so we don't have to re-run the full pipeline every time we want to analyse it.
#
# epochs: mne.Epochs — the processed epoch object to save
# out_path: str      — the file path to save to (e.g. "data/processed/sub-01-epo.fif")
def save_processed(epochs: mne.Epochs, out_path: str) -> None:
    # Create the output directory if it doesn't exist yet.
    # parents=True creates intermediate folders (like mkdir -p).
    # exist_ok=True means no error if the folder already exists.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # epochs.save writes the MNE Epochs object to disk in .fif format.
    # overwrite=True replaces an existing file without asking —
    # safe here because we always regenerate from raw anyway.
    epochs.save(out_path, overwrite=True)
```

### 5.3 Save the file
Press `Ctrl + S`

### 5.4 Verify pipeline.py still imports cleanly
```powershell
python -c "from src.preprocessing.filter import load_raw, apply_filters; print('filter.py OK')"
```
**Expected output:**
```
filter.py OK
```

---

## PART 6 — First Two Commits (10 min)

### 6.1 Stage and commit loader.py
```powershell
git add src/preprocessing/loader.py
git commit -m "M1: add loader.py with load_raw and inspect_raw"
```
- `git add` = stage the file (tell git "include this in the next commit")
- `git commit -m` = create the commit with the message in quotes

**Expected output:**
```
[feature/M1-data-loading xxxxxxx] M1: add loader.py with load_raw and inspect_raw
 1 file changed, 16 insertions(+)
```

### 6.2 Stage and commit the updated filter.py
```powershell
git add src/preprocessing/filter.py
git commit -m "M1: remove load_raw from filter.py, import from loader"
```
**Expected output:**
```
[feature/M1-data-loading xxxxxxx] M1: remove load_raw from filter.py, import from loader
 1 file changed, 3 insertions(+), 4 deletions(-)
```

### 6.3 Verify both commits exist
```powershell
git log --oneline
```
**Expected output** (newest first):
```
xxxxxxx M1: remove load_raw from filter.py, import from loader
xxxxxxx M1: add loader.py with load_raw and inspect_raw
xxxxxxx Initial commit
```

---

## PART 7 — Start the M1 Notebook (70 min)

### 7.1 Launch JupyterLab
In the terminal:
```powershell
jupyter lab
```
A browser tab opens automatically at `http://localhost:8888`. If it doesn't, copy the URL shown in the terminal and paste it into your browser.

### 7.2 Create the notebook
- In JupyterLab, look at the left panel — you'll see the project folder
- Click into the `notebooks/` folder
- Click the `+` button (New Launcher) → **Python 3 (ipykernel)**
- A new untitled notebook opens
- Rename it: right-click the tab → **Rename** → `M1_data_loading.ipynb`

---

### 7.3 Cell 1 — Imports

Type this in the first cell:
```python
import mne
# mne is the entire EEG/MEG analysis library.
# Importing it gives us access to Raw, Epochs, Evoked objects
# and all processing functions used across M1–M9.

from src.preprocessing.loader import load_raw, inspect_raw
# We import our own functions from loader.py —
# the file we just built. This proves the module works
# and lets us use it interactively in the notebook.
```
Run it: `Shift + Enter`

**Expected output:** No output, no errors.

---

### 7.4 Cell 2 — Download and load the MNE sample dataset

```python
# mne.datasets.sample.data_path() checks if the MNE sample dataset
# is already downloaded on this machine. If yes, returns the path.
# If no, downloads it automatically (~1.5 GB, takes 5–10 min first time).
# The dataset contains auditory and visual EEG+MEG recordings
# from one subject — standard demo data used across all MNE tutorials.
sample_path = mne.datasets.sample.data_path()

# Build the full path to the raw .fif file inside the dataset folder.
# The / operator on Path objects joins folder names — works on Windows and Mac.
# "MEG/sample/sample_audvis_raw.fif" is where the raw recording lives
# inside the downloaded dataset.
raw_path = sample_path / "MEG" / "sample" / "sample_audvis_raw.fif"

# Call our load_raw function from loader.py.
# str() converts the Path object back to a string because
# load_raw expects a string argument (which it then converts to Path internally).
# This gives us back an mne.io.Raw object called `raw` that holds
# the full 277-second recording in memory.
raw = load_raw(str(raw_path))
```
Run it: `Shift + Enter`

**Expected output:**
```
Opening raw data file ...
    Read a total of 3 projection items:
        PCA-v1 (1 x 102)  idle
        ...
Ready.
```
First run downloads data and takes a few minutes. Subsequent runs are instant.

---

### 7.5 Cell 3 — Inspect the raw object

```python
# Call our inspect_raw function from loader.py.
# This prints the four key metadata fields:
# channel count, sampling rate, duration, and channel types.
# Always run this after loading — it confirms the file loaded
# correctly and tells you what you're working with before any analysis.
inspect_raw(raw)
```
Run it: `Shift + Enter`

**Expected output:**
```
Channels    : 376
Sampling rate: 600.614990234375 Hz
Duration    : 277.0 seconds
Channel types: {'grad', 'mag', 'eeg', 'stim', 'eog'}
```

What each line tells you:
- **376 channels** — this dataset has EEG + MEG sensors combined. Later in M9 we switch to a pure 64-channel EEG dataset.
- **600 Hz** — 600 samples per second per channel. High enough to resolve fast neural events.
- **277 seconds** — about 4.5 minutes of recording.
- **Channel types** — `grad` and `mag` are MEG sensors, `eeg` is EEG, `stim` is the trigger channel that marks when stimuli happened, `eog` is the eye movement channel used later in M5 ICA.

---

### 7.6 Cell 4 — Plot the raw signal

```python
# raw.plot() opens an interactive browser showing the raw signal.
# duration=5   — show 5 seconds of data at a time in the window
# n_channels=20 — show 20 channels stacked vertically at once
# title=...    — label shown at the top of the plot window
#
# What you will see:
# - Slow wavy drifts (low-frequency noise below 1 Hz — removed in M2 by high-pass filter)
# - Sharp spikes (muscle or eye movement artifacts — removed in M4 and M5)
# - Some channels that look flat or dead (bad channels — marked in M4)
# - A generally noisy, hard-to-read signal — this is exactly why preprocessing exists
raw.plot(duration=5, n_channels=20, title="Raw unprocessed signal — first 5 seconds")
```
Run it: `Shift + Enter`

A scrollable plot window will open. Scroll through it and observe the signal before closing.

---

### 7.7 Cell 5 — Markdown observation (change cell type to Markdown first)

Click `+ Code` to add a new cell. Then click the dropdown at the top that says **Code** and change it to **Markdown**.

Type this — fill in the brackets with what you actually saw in the plot:
```markdown
## What I Observed

The raw signal has 376 channels recorded at ~600 Hz over ~277 seconds.
Channel types include EEG, MEG gradiometers, MEG magnetometers,
a stimulus channel, and EOG.

In the 5-second plot:
- [Describe what you saw — slow drifts? spike artifacts? flat channels?]
- [Did certain channels look noisier than others?]
- [What did the EOG channel look like compared to EEG channels?]

This is why preprocessing exists — the raw signal is not usable
directly for analysis. Each module (M2–M5) removes one category of noise.
```
Run it: `Shift + Enter` (renders the markdown)

---

## PART 8 — Third Commit (5 min)

Go back to the VS Code terminal (keep JupyterLab open in the browser).

```powershell
# Stage the notebook file
git add notebooks/M1_data_loading.ipynb

# Commit with a descriptive message
git commit -m "M1: notebook draft — load MNE sample, inspect raw, plot signal"
```

### Final check — confirm all three commits
```powershell
git log --oneline
```
**Expected output:**
```
xxxxxxx M1: notebook draft — load MNE sample, inspect raw, plot signal
xxxxxxx M1: remove load_raw from filter.py, import from loader
xxxxxxx M1: add loader.py with load_raw and inspect_raw
xxxxxxx Initial commit
```

---

## END OF SESSION CHECKLIST

Before closing everything:

- [ ] `(.venv)` was active the whole session
- [ ] `src/preprocessing/loader.py` exists with `load_raw` and `inspect_raw`
- [ ] `filter.py` no longer defines `load_raw` — it imports it from `loader.py`
- [ ] `notebooks/M1_data_loading.ipynb` has 5 cells (imports, load, inspect, plot, markdown)
- [ ] 3 commits on branch `feature/M1-data-loading`
- [ ] The raw plot showed actual data (not an error)

If all boxes are checked — session complete. Friday (7 hrs) you finish the notebook and merge the PR.

---

## IF SOMETHING GOES WRONG

**`.venv\Scripts\activate` not recognized**
- Make sure you typed a single dot: `.venv` not `..venv`
- Run `pwd` first to confirm you are in `C:\Users\ke725\Documents\eeg-project`

**`ModuleNotFoundError: No module named 'src'`**
- Your Jupyter kernel is not running from the project root
- In the terminal: `cd C:\Users\ke725\Documents\eeg-project` then relaunch `jupyter lab`

**`ModuleNotFoundError: No module named 'mne'`**
- The venv is not activated
- Run `.venv\Scripts\activate` in the terminal, then relaunch `jupyter lab`

**MNE sample download is stuck**
- Let it run — the first download takes 5–10 min depending on your internet
- Do not cancel it mid-download

**`git commit` opens a text editor unexpectedly**
- Type your commit message, then press `Ctrl + X`, then `Y`, then `Enter` to save and exit
