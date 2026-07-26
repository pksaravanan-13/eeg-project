import mne


def fit_ica(epochs: mne.Epochs, n_components: int = 20, random_state: int = 42) -> mne.preprocessing.ICA:
    """Fit an ICA decomposition to epoched EEG data.

    Separates the signal into statistically independent components so
    artifact sources (eye blinks, muscle, cardiac) can later be removed
    surgically via apply_ica(), instead of discarding whole trials.

    Args:
        epochs: Epochs to fit on -- should already have the worst
            whole-trial contamination removed (reject_by_amplitude())
            first, so the decomposition isn't spent fitting components
            to trials that were always getting thrown out.
        n_components: Number of components to fit. Defaults to 20:
            roughly a third of this pipeline's typical ~60 EEG channels
            -- enough to separate blink/muscle/cardiac from brain
            activity without also chasing pure sensor noise into its
            own component.
        random_state: Seed for reproducibility. Defaults to 42 because
            ICA's optimization starts from a random point and can
            otherwise converge to differently-ordered components on
            every run, breaking reproducibility.

    Returns:
        The fitted ICA object (not yet applied to any data).
    """
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
    """Identify which ICA components correlate with eye-blink activity.

    Args:
        ica: A fitted ICA object (see fit_ica()).
        epochs: The same epochs ica was fit on. Must still contain the
            EOG channel -- find_bads_eog() correlates each component's
            time course against it.

    Returns:
        Indices of components that correlate with the EOG channel and
        should be excluded via apply_ica(). Does not return the
        underlying correlation scores -- this function's contract is
        just "which components to exclude"; call find_bads_eog()
        directly if the scores are needed for a diagnostic plot.
    """
    eog_indices, eog_scores = ica.find_bads_eog(epochs)

    print(f"Auto-detected EOG component(s): {eog_indices}")

    return eog_indices


def apply_ica(ica: mne.preprocessing.ICA, epochs: mne.Epochs, exclude: list) -> mne.Epochs:
    """Remove the given ICA components from epochs, returning a cleaned copy.

    Args:
        ica: A fitted ICA object (see fit_ica()).
        epochs: The epochs to clean. Left unmodified -- a copy is
            returned so a before/after comparison stays possible.
        exclude: Component indices to remove, e.g. from
            auto_detect_eog(). Passed explicitly rather than hardcoded
            so a manual override after visual inspection doesn't
            require touching this function.

    Returns:
        A copy of epochs with the excluded components' contribution
        subtracted out.
    """
    ica.exclude = exclude
    epochs_clean = epochs.copy()
    ica.apply(epochs_clean)
    return epochs_clean
