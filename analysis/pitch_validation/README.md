# Preliminary pitch validation

> Historical experiment record. Redundant audio and exploratory demo scripts were removed during cleanup. See the [current repository instructions](../../README.md) for retained data and supported regeneration/evaluation commands.

For the completed dataset-scale PSOLA normalization and matched HuBERT results, see [PSOLA_EXPERIMENT.md](PSOLA_EXPERIMENT.md).

For the actor-level batch normalization, ±4-semitone cap, generated datasets, and matched HuBERT results, see [ACTOR_PSOLA_EXPERIMENT.md](ACTOR_PSOLA_EXPERIMENT.md).

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

## Controlled quality comparison

After the investigation above, run:

```bash
python src/compare_shift_quality.py
```

This creates `quality_comparison/` within the Actor 3 investigation directory. The original is 48 kHz PCM16; the saved normalized file is 16 kHz FLOAT. All listening files in this comparison are 16 kHz PCM24, with RMS matched over the same original-derived active time blocks and a shared peak-safety scale. This helps separate pitch-processing differences from reduced bandwidth or volume; it is not perceptual loudness matching. Numerical measurements use the waveforms before these listening gains.

`listening_sequence.wav` plays the following clips in order, with 0.75 seconds of silence between them. Separate numbered WAV files are also provided:

1. Original resampled to 16 kHz, without pitch shifting.
2. Historical saved normalized recording (its applied shift remains unknown).
3. Fresh default Librosa normalization.
4. Same requested shift with a 512-sample STFT window and 128-sample hop at 16 kHz.
5. Same requested shift at the original 48 kHz with default STFT parameters, followed by resampling to 16 kHz.

The current requested shift is +2.2875 semitones in conditions 3–5. The default shifter uses a 2,048-sample STFT window: 128 ms at 16 kHz versus 42.7 ms at 48 kHz; condition 4 uses 32 ms at 16 kHz. The estimator configuration stays fixed across comparisons. On this single recording, pYIN's output means were 411.90, 205.09, and 210.99 Hz for conditions 3–5. A repeated default shift exactly reproduced the fresh waveform (maximum absolute difference 0), so there is no evidence of random variation in that rerun. Historical environment or processing differences remain unresolved.

Twenty of the saved file's 21 additional voiced detections were below -40 dB relative to its loudest RMS frame; ten of the fresh file's 16 additional detections were below that threshold. These are quiet-frame diagnostics, not verified labels of silence or ground-truth pitch errors. `manifest.json` records the measurements and gains, while `spectrograms.png` uses a shared magnitude reference across variants. No subjective listening verdict is assigned by the script, and neither experimental condition is a validated replacement.

As a separate diagnostic, normalized autocorrelation over 2.46–2.56 seconds has its strongest candidate near 207.79 Hz in the original and 235.29 Hz in the fresh default output. The latter also has a strong candidate at 470.59 Hz, consistent with ambiguity between the fundamental and its second harmonic. The manifest records several candidates for every variant. This short-excerpt check supports an octave-error hypothesis for pYIN in that region, but is not ground truth or a recording-wide pitch measurement.

Librosa documents its [phase vocoder](https://librosa.org/doc/0.11.0/generated/librosa.phase_vocoder.html) as a simplified implementation that does not handle transients and can produce audible artifacts. Candidate follow-up engines include [Rubber Band](https://breakfastquay.com/rubberband/integration.html), with its higher-quality engine and optional formant preservation, and [Praat's speech overlap-add synthesis](https://praat.org/manual/overlap-add.html). These have not been installed or evaluated in this comparison.

## Candidate solutions and alternative-engine listening test

The user preferred clips 4 and 5 in the quality comparison above. Retain the shorter-window and native-rate Librosa approaches together as **solution 1**, pending broader evaluation. This is subjective preference on the inspected sample, not evidence of improved recognition accuracy.

`src/compare_pitch_engines.py` adds **solution 2: Rubber Band R3**, with formant preservation off/on, and **solution 3: Praat TD-PSOLA**, via Parselmouth. Praat locates pitch cycles and uses overlap-add synthesis; Rubber Band is a more sophisticated time/pitch-processing engine. These examples were generated after the preceding quality comparison and are stored separately in `pitch_engine_comparison_03-01-03-01-01-01-03/`.

For the local exploratory run, `praat-parselmouth==0.4.7` was installed in `/private/tmp/ser-pitch-engines`, exposing Praat 6.1.38. The official Rubber Band 4.0.0 macOS command-line archive was downloaded from Breakfast Quay and extracted into `/private/tmp/rubberband-4.0.0-gpl-executable-macos/`. The existing Conda environment was not modified. These temporary dependencies may need to be restored if the OS clears temporary storage. With those paths available, run from the repository root:

```bash
conda activate speech-emotion
PYTHONPATH=/private/tmp/ser-pitch-engines python src/compare_pitch_engines.py --rubberband-path /private/tmp/rubberband-4.0.0-gpl-executable-macos/rubberband
```

For an ordinary installation, expose `praat-parselmouth` to the Python interpreter and provide the Rubber Band executable through `PATH` or `--rubberband-path`. `--file FILENAME.wav` selects another RAVDESS recording, and `--semitones 2 5` selects fixed test shifts. No training-data generation defaults are changed.

The default test creates groups at +2.2875 semitones (the existing formula's shift for Actor 3) and +5 semitones (a larger upward shift, not a 200-Hz normalization condition). Each group's RMS-matched, 16-kHz `listening_sequence.wav` contains six clips separated by 0.75 seconds:

1. Original, unshifted, resampled to 16 kHz.
2. Solution 1: Librosa's shorter window at 16 kHz.
3. Solution 1: Librosa at the original 48 kHz, then resampled to 16 kHz.
4. Rubber Band 4.0.0, offline R3 engine, without formant preservation.
5. Rubber Band 4.0.0, offline R3 engine, with formant preservation (`-F`).
6. Praat pitch-tier multiplication and overlap-add resynthesis, performed at 48 kHz and then resampled to 16 kHz. Pitch analysis uses a 0.01-second step, 50-Hz floor and 800-Hz ceiling.

All shifted clips within a group request exactly the same pitch ratio. Native-rate results and unadjusted 16-kHz results are in `raw/`; top-level numbered WAVs include only the listening gain. The manifest records package versions, Rubber Band's executable hash/commands, listening gains, sample-count differences, pYIN statistics, and a separate Praat autocorrelation pitch estimate. The original pYIN and Praat own-frame means differ (175.25 vs 186.83 Hz), illustrating that neither is ground truth. No subjective ranking is assigned to the new engines. All generated native outputs retained the original sample count, and all fourteen listening WAVs were checked for 16-kHz sample rate, finite samples and peaks below full scale.

For the +2.2875-semitone condition, the pYIN own-frame means were 205.09 Hz (short-window Librosa), 210.99 Hz (native-rate Librosa), 316.73 Hz (Rubber Band without formant preservation), 254.02 Hz (Rubber Band with formant preservation), and 209.16 Hz (Praat). Median measured shifts on shared voiced frame positions were approximately +2.3 semitones for all five, and approximately +5.0 in the larger-shift condition. These diagnostics do not rank perceptual quality or establish recording-wide true F0.

`zero_shift_controls/` retains the original waveform and resynthesis outputs with a requested zero-semitone shift. These help investigate coloration caused by processing alone; waveform differences, particularly in pitch-synchronous resynthesis, are not perceptual quality scores.

Relevant primary references: [DigiTech Drop](https://digitech.com/dp/drop/) describes polyphonic drop tuning but does not disclose implementation details on the product page; [Rubber Band CLI](https://github.com/breakfastquay/rubberband/blob/default/main/main.cpp) documents the R3 and formant options; [Parselmouth pitch manipulation](https://parselmouth.readthedocs.io/en/stable/examples/pitch_manipulation.html) demonstrates the pitch-tier/overlap-add workflow used here.
