# Preliminary pitch validation

These files preserve the diagnostic measurements used in the midterm report's preliminary finding. They are separate from the training and evaluation outputs.

- `sample_measurements.csv` and `sample_measurements.json`: one normal-intensity happy recording per actor, statement 1, repetition 1; 24 original/normalized pairs. The JSON also records voiced-frame counts, median F0 and mean voicing probability over the selected frames.
- `matched_frames.json`: additional checks for actors 1, 2, 3, 5, 11 and 21, restricted to frame positions identified as voiced in both versions.

## Method

Original files were read from `data/RAVDESS/Actor_XX/` and saved processed files from `data/audio_variants/normalized_only/Actor_XX/`, using identical filenames. Both versions were loaded as mono at 16 kHz using Librosa 0.11.0. F0 was estimated with `librosa.pyin(audio, sr=16000, fmin=50, fmax=800)`, using its default 2,048-sample frame and 512-sample hop. A valid frame was marked voiced and had a finite F0. Each recording's mean was the arithmetic mean over its valid frames, matching the repository's normalization statistic.

For the matched-frame check, means were recomputed over the intersection of valid frame positions in the original and processed recordings. This examines the same nominal times in both versions, but does not establish ground-truth voicing or pitch.

## Example quoted in the report

Recording `03-01-03-01-01-01-03.wav`:

| Measurement | Original | Saved normalized |
|---|---:|---:|
| Mean F0 over each version's own voiced frames | 175.25 Hz | 272.55 Hz |
| Number of those voiced frames | 36 | 56 |
| Mean F0 over the 35 shared voiced frame positions | 175.97 Hz | 205.41 Hz |

## Interpretation limits

The estimator is the same family and configuration used in the repository, so these are diagnostic estimates rather than independent ground truth. Different voiced-frame selections can change the measured mean even if a shift behaves as intended. Historical package versions and per-recording shifts were not saved with the generated dataset; these measurements cannot reconstruct its exact generation environment. The `formula_shift_semitones` column is computed from the audit's input estimate, not a recovered record of the shift originally applied. The sample is not a full-dataset audit, and these measurements do not establish why recognition performance decreased.

## Reproduce and extend the Actor 3 investigation

From the repository root:

```bash
conda env list
conda activate speech-emotion
python src/investigate_pitch.py
```

The script calls the existing `normalize_f0()` without changing it. It compares the original recording, the saved normalized recording, and freshly normalized audio saved as FLOAT WAV and loaded again. Results go to `analysis/pitch_validation/investigation_03-01-03-01-01-01-03/`: package versions and measurements in `summary.json`, frame positions/F0/voicing probabilities in `frames.csv`, a `pitch_tracks.png` plot, and `fresh_normalized.wav`. Another recording can be selected with `--file FILENAME.wav`; `--output-dir PATH` changes the artifact directory. Diagnostic caches use temporary writable directories when cache environment variables are not already set.

Initial run on 2026-11-12 (Python 3.11.17, Librosa 0.11.0):

| Measurement | Saved normalized | Fresh normalized |
|---|---:|---:|
| Mean F0 over each output's voiced frames | 272.55 Hz | 411.90 Hz |
| Shared-frame output mean F0 | 205.41 Hz | 239.29 Hz |
| Newly voiced frames relative to original | 21 | 16 |
| Mean F0 of newly voiced frames | 384.45 Hz | 713.97 Hz |
| Mean pYIN voicing probability of newly voiced frames | 0.0102 | 0.0101 |
| Median shift over shared frames | +2.40 semitones | +2.30 semitones |

The current formula requests +2.2875 semitones. The saved file's actual historical requested shift is unknown. Each output has its own shared-frame selection, so the two shared means above are not measured on identical subsets. The fresh output also contains shared-frame shifts near one octave above the requested shift; its median alone does not characterize every frame. The low-probability new detections and octave-separated estimates motivate spectral/audio inspection; these results do not independently establish whether pYIN or the transformation is responsible. The saved and fresh waveforms differ, and the cause of that historical difference has not been established.
