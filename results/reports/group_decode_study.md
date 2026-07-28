# Group Decode Study: Does ICA-Based Artifact Removal Improve Decode Accuracy?

**Hypothesis:** ICA-based artifact-component removal (EOG-proxy correlation, since PhysioNet
EEGBCI ships no true EOG channel) improves mu/beta band-power decode accuracy relative to no
cleaning at all, across subjects.

**Note on scope:** this study omits config.yaml's amplitude-rejection stage (150uV
peak-to-peak) for this dataset specifically -- confirmed empirically that threshold drops
100% of trials here, since it was tuned against the sample dataset's ~1s auditory epochs, and
peak-to-peak amplitude over EEGBCI's much longer 5s imagery window exceeds it on every trial.
A discovered limitation of a fixed-threshold criterion across differing epoch durations, not
a bug worked around silently.

**Subjects:** 15 (PhysioNet EEGBCI, IDs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

**Mean cross-validated accuracy:**
- No cleaning (baseline): 0.550
- ICA-cleaned: 0.532

**Paired Wilcoxon signed-rank test:** statistic=31.500, p=0.3270

**Interpretation:** Across this sample, ICA-based cleaning does not improve mean decode accuracy
(p=0.3270). This does not clear the conventional 0.05 threshold -- read as a null result, not a negative one, given the small subject count.
