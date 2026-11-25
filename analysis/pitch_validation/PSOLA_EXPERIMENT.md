# PSOLA normalization and matched HuBERT comparison

> Historical experiment record. Redundant audio and exploratory demo scripts were removed during cleanup. See the [current repository instructions](../../README.md) for retained data and supported regeneration/evaluation commands.

## Question and condition

The user preferred Praat PSOLA in the +2.2875- and +5-semitone listening comparisons. This experiment tests whether substituting that pitch shifter improves speech-emotion recognition relative to the saved Librosa-normalized audio. The shorter-window/native-rate Librosa approaches remain solution 1; this experiment does not select them as the production default.

The new `normalized_psola` condition keeps the existing utterance-wise arithmetic mean-F0 target of 200 Hz. The original is resampled to 16 kHz solely for pYIN estimation (50–800 Hz, 2,048-sample frame, 512-sample hop). The requested shift is `12 * log2(200 / mean_f0)`. Praat PSOLA applies that fixed ratio at the original 48 kHz, using a 0.01-second analysis step and the same 50–800-Hz search limits; the result is then resampled to 16 kHz and saved as FLOAT WAV. No listening gain, noise reduction, shift cap, repeated corrective shifting, or augmentation is applied.

This changes both the synthesis method and the sampling rate at which synthesis operates relative to the old 16-kHz Librosa pipeline. It tests the complete PSOLA preprocessing candidate, not a decomposition of those two factors. pYIN's input estimator is retained; unreliable estimates and extreme requested shifts remain possible. Output pYIN measurements are diagnostics, not independent ground truth. The saved-output audit reads FLOAT WAV as float64; downstream feature extraction uses Librosa's float32 loading.

The historical dataset did not record per-file shifts or package versions. Reusing its saved waveforms therefore does not ensure identical historical pitch estimates or requested shifts to the current run. This comparison can establish the relative performance of these available pipelines on a matched cohort; it cannot attribute every difference specifically to PSOLA versus Librosa synthesis. A future engine-isolation experiment should generate both conditions from the same recorded shifts and native-rate inputs.

## Dataset

Generation produced 1,436 recordings from 1,440 inputs. Four had fewer than three valid pYIN voiced frames and were excluded; no additional recordings were rejected by Praat. The saved historical Librosa condition has 1,432 recordings and a different exclusion set. The intersection across original, historical Librosa, and PSOLA contains **1,429 recordings**. All three models are trained, validated, and tested using only that shared set.

Generation logs, applied shifts, original/output F0 diagnostics, file hashes, versions, and exclusions are stored in `data/audio_variants/normalized_psola/`. The generator verifies each saved WAV against its float32 samples. It resumes using source/output hashes and rejects changed generation configurations. The existing `normalized_only` dataset is retained.

## Evaluation

The comparison uses frozen `facebook/hubert-base-ls960` final-layer features, averaged over time, followed by a newly trained 768-to-8 linear classifier in each fold. The pretrained snapshot is `dba3bb02fda4248b6e082697eee756de8fe8aa8a`. Feature extraction uses the Mac GPU; classifiers train on CPU. Equal-length recordings are batched without padding, preserving the original mean-pooling semantics. Embeddings are cached with source hashes and an environment/model signature.

Each condition uses the existing `output/splits.csv`: 24 leave-one-speaker-out folds with 21 training actors, two validation actors, and one test actor. Training uses seeds 42, 123, and 999, 30 epochs, batch size 32, and Adam at learning rate 0.001. The epoch with the highest validation UAR supplies the test checkpoint. For each seed, predictions from the 24 test folds are pooled before calculating UAR/WAR. Summary standard deviations are sample standard deviations over three seed-level scores.

These are fresh matched reruns in the current environment, not direct replacements for the historical report's unmatched scores. Original classifier checkpoint files in the checkout were Git LFS pointers, so historical trained classifiers were not reused. Output files live under `output/pitch_comparison_hubert/{original,normalized_only,normalized_psola}/hubert_baseline/`; `comparison.csv`, `matched_recordings.csv`, and `comparison_config.json` record the comparison and its exact cohort/settings.

## Reproduction

The exploratory Parselmouth installation is in `/private/tmp/ser-pitch-engines`; it may disappear if temporary storage is cleared. For a persistent setup, install the optional `requirements-pitch.txt` into the project environment. The public pretrained HuBERT snapshot is cached in the repository's ignored `.cache/huggingface/`, with its path in `.cache/hubert_model_path.txt`.

```bash
conda activate speech-emotion
PYTHONPATH=/private/tmp/ser-pitch-engines python src/create_psola_variants.py --workers 4
python src/evaluate_pitch_conditions.py
```

With a persistent Parselmouth installation, the `PYTHONPATH` prefix is unnecessary. On a Mac, run the evaluation outside a sandbox that blocks Metal access to enable `mps`; `--device cpu` is supported. All three datasets must have completed indices before evaluation. The evaluation script refuses to reuse an output root if its cohort, environment, folds, seeds, or model snapshot changes.

## Verification performed

- Four-recording generation smoke run completed and verified saved audio.
- Repeating the smoke run reused all four verified outputs without regenerating audio.
- The shared PSOLA helper exactly reproduced the direct Praat calls used in the preferred listening test; silent input was explicitly rejected.
- All 24 matched splits were checked for recording coverage and disjoint actor sets (21/2/1).
- Full generation completed with explicit exclusions, finite saved samples, and unchanged sample counts.

## Results (2026-11-12)

All 216 condition/seed/fold training runs completed. Each condition has 4,287 held-out predictions: exactly one prediction per matched recording per seed. Metrics were independently recomputed from the saved predictions, including sample standard deviations, and checked against `evaluation.json`.

| Audio condition | UAR (%) | WAR / accuracy (%) |
|---|---:|---:|
| Original | 58.63 ± 0.79 | 59.60 ± 0.51 |
| Historical Librosa normalization | 42.81 ± 1.52 | 43.92 ± 1.33 |
| New Praat PSOLA normalization | 55.85 ± 1.08 | 56.68 ± 0.87 |

Values are mean ± sample standard deviation over the three seeds. PSOLA exceeded historical Librosa normalization by **13.04 UAR percentage points**, while remaining **2.78 points below original audio**. The paired seed-level gains were +12.60, +12.40, and +14.11 points for seeds 42, 123, and 999 respectively. No statistical significance claim is made from these three training seeds.

| Emotion | Original mean recall (%) | Historical Librosa (%) | PSOLA (%) |
|---|---:|---:|---:|
| Angry | 74.00 | 53.40 | 70.51 |
| Calm | 82.81 | 72.40 | 77.26 |
| Disgust | 68.26 | 44.50 | 66.31 |
| Fearful | 68.58 | 61.28 | 64.58 |
| Happy | 41.67 | 19.10 | 41.32 |
| Neutral | 44.10 | 26.39 | 43.40 |
| Sad | 27.72 | 18.60 | 27.72 |
| Surprised | 61.88 | 46.81 | 55.67 |

The result supports the PSOLA pipeline as a substantially better normalization candidate than the saved historical pipeline in this experiment. It does not establish an advantage over using original audio, and it does not isolate synthesis artifacts as the sole cause of the historical degradation. The retained mean-F0 estimator, potentially large shifts (observed range approximately -23.91 to +24.00 semitones), removal of emotion-related average-pitch differences, and historical generation differences remain unresolved factors. No shift clipping or post-hoc condition tuning was performed after observing these test scores.

Machine-readable results are in `output/pitch_comparison_hubert/comparison.csv`, `seed_scores.csv`, `per_emotion_seed_recall.csv`, and `per_emotion_mean_recall.csv`. Full predictions, histories, validation-selected classifier checkpoints, and fold metrics are preserved for each condition.
