import mne


def fit_ica(epochs: mne.Epochs, n_components: int = 20, random_state: int = 42) -> mne.preprocessing.ICA:
    # n_components=20: roughly a third of this dataset's ~60 EEG channels —
    # enough to separate blink/muscle/cardiac from brain activity without
    # also chasing pure sensor noise into its own component. random_state=42
    # makes the fit reproducible: ICA's optimization starts from a random
    # point and can otherwise converge to differently-ordered components on
    # every run.
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter="auto",
    )
    # picks="eeg": the EOG channel must stay untouched by the decomposition
    # so auto_detect_eog() has an uncontaminated reference to correlate against.
    ica.fit(epochs, picks="eeg")
    return ica


def auto_detect_eog(ica: mne.preprocessing.ICA, epochs: mne.Epochs) -> list:
    # epochs must still contain the EOG channel — find_bads_eog correlates
    # each component's time course against it.
    eog_indices, eog_scores = ica.find_bads_eog(epochs)

    print(f"Auto-detected EOG component(s): {eog_indices}")

    return eog_indices
    # eog_scores isn't returned — this function's contract is just "which
    # components to exclude." Call find_bads_eog() again directly if the
    # scores are needed for a diagnostic plot.


def apply_ica(ica: mne.preprocessing.ICA, epochs: mne.Epochs, exclude: list) -> mne.Epochs:
    # exclude is passed explicitly (not hardcoded) so a manual override
    # after visual inspection doesn't require touching this function.
    ica.exclude = exclude

    # Never mutate in place — keeps a before/after pair for comparison plots.
    epochs_clean = epochs.copy()

    ica.apply(epochs_clean)
    return epochs_clean
