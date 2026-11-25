# Active Praat actor-median normalization experiment

Date: 2026-11-23.

Both new conditions improved over their corresponding old actor-mean conditions in all three seeds. The capped new condition had the highest observed average UAR, with a small, seed-dependent gain over original audio.

## Recognition results

| Condition | UAR (%) | WAR / accuracy (%) |
|---|---:|---:|
| Original audio | 58.63 ± 0.79 | 59.60 ± 0.51 |
| Old actor mean, full shift | 56.50 ± 0.77 | 57.27 ± 0.45 |
| Old actor mean, ±4-ST cap | 56.95 ± 0.37 | 58.01 ± 0.19 |
| New active Praat median, full shift | 58.54 ± 0.62 | 59.32 ± 0.39 |
| New active Praat median, ±4-ST cap | **59.06 ± 0.62** | **59.79 ± 0.63** |

Mean ± sample standard deviation over three seeds; each seed pools predictions across the 24 held-out-speaker folds. All conditions use exactly 1,429 recordings. Historical evaluations were reused only after matching their settings/cohort and verifying that current WAV hashes match their feature manifests. All reported metrics were independently recomputed from the saved predictions.

- New full versus old full: **+2.05 UAR percentage points**, with paired seed gains +2.19, +2.30 and +1.65.
- New capped versus old capped: **+2.12 points**, with paired seed gains +1.90, +2.81 and +1.64.
- New full versus original: **-0.08 points** on average; paired differences -0.39, -0.19 and +0.33.
- New capped versus original: **+0.44 points** on average; paired differences -0.27, +1.44 and +0.13. It exceeds original in two of three seeds, so this is not a consistent or statistically established improvement over original audio.

The result supports the new procedure over the crude actor-mean normalization in this matched frozen-HuBERT experiment. It brings full normalization close to original-audio recognition, and capped normalization is a promising candidate. It does not establish a general advantage across models, datasets or deployment settings. The two caps were specified before seeing these results; no further test-score tuning was performed.

See the [recognition comparison figure](praat_median_experiment/recognition_comparison.png), [all seven conditions](../../output/pitch_comparison_praat_median_hubert/comparison_all_conditions.csv), [paired seed comparisons](../../output/pitch_comparison_praat_median_hubert/paired_comparisons.json), and [independent result validation](../../output/pitch_comparison_praat_median_hubert/results_validation.json). Historical utterance-level PSOLA and Librosa conditions remain in the full comparison table.

## Protocol

This experiment tests the robust baseline candidate from [the F0 audit](F0_AUDIT.md). The candidate uses Praat 7.0.03 filtered autocorrelation, active original-audio frames at least -35 dB relative RMS, and selected-candidate strength >=0.6. Each utterance supplies its median log F0; the actor baseline is the median of the 60 utterance medians. All recordings have at least ten selected frames. No emotion labels enter the fitted baseline.

The baseline is recomputed from hash-verified original audit tracks and checked against the reported audit table. All 24 actors are retained under one rule, including those with audit review flags. There is no actor-specific adjustment, target tuning, or selection of settings based on recognition test scores.

Two conditions were specified before evaluating recognition:

- `data/audio_variants/normalized_psola_actor_praat_median/`: one fixed full actor shift, `12 * log2(200 / actor_baseline_hz)`.
- `data/audio_variants/normalized_psola_actor_praat_median_cap4/`: the same shift clipped to [-4, +4] semitones, matching the earlier capped condition.

Full shifts range from -6.673 to +12.194 ST. Fourteen actors request shifts beyond four semitones; the capped condition's theoretical centres consequently span 124.59–233.39 Hz rather than all reaching 200 Hz.

Synthesis is the **unchanged** previous native-rate PSOLA helper, using Parselmouth 0.4.7 / embedded Praat 6.1.38, a 10-ms step and 50–800-Hz pitch limits, followed by 16-kHz resampling and FLOAT-WAV export. Keeping the old synthesis implementation isolates the baseline-estimation change. Current Praat supplies the baseline and output measurements, not a replacement synthesis engine. No gain matching is applied to training audio.

The comparison changes the estimator, selected frames, and summary statistic together. It tests this complete baseline procedure; it does not isolate each component's causal contribution.

Like the previous actor experiment, this is **batch speaker adaptation**: an actor's available recordings contribute to that actor's baseline, including held-out-speaker recordings. It does not use their emotion labels, but it does assume access to the speaker batch. It is not evidence for first-utterance deployment.

## Verification and evaluation

The generator records source/output hashes, baseline provenance, versions, actor shifts and flags, and exclusions. It verifies decoded saved samples, sample rate, finite values and duration. A 24-actor smoke run exercises both conditions and the full shift range; a repeat run reused all 48 verified files without synthesis.

All generated outputs are independently remeasured with current Praat filtered autocorrelation. The delivered ratio is compared to the request on original-active, nominally aligned frames with strong input and output estimates. Measured actor centres use the same median-of-utterance-medians procedure. Changed voicing, finite analysis bounds, detector errors and local PSOLA timing remain limitations; these are diagnostic estimates rather than ground-truth measurements.

Recognition follows the previous frozen final-layer HuBERT features plus linear-classifier experiment: checkpoint `facebook/hubert-base-ls960`, snapshot `dba3bb02fda4248b6e082697eee756de8fe8aa8a`, mean time pooling, 768-to-8 linear heads, Adam at 0.001, batch size 32, 30 epochs. Each fold's checkpoint is selected by validation UAR. There are 24 LOSO folds (21 training, two validation and one test actor) and seeds 42, 123 and 999.

Use the established **1,429-recording matched cohort**, even if the new generators produce additional recordings. That keeps the comparison to original, utterance-normalized and old actor-normalized audio matched. The original/old conditions reuse completed evaluations. A separate scoring script checks equal settings, exact test-split/cohort coverage, 4,287 unique held-out predictions per condition, and independently reproduces every UAR and WAR summary from predictions. Standard deviations are sample standard deviations across three seed-level pooled scores. These are observed differences, not a statistical significance claim.

Training and comparison artifacts live under `output/pitch_comparison_praat_median_hubert/`. Acoustic validation artifacts live under `analysis/pitch_validation/praat_median_experiment/`.

## Generation and acoustic results

Each new folder contains **all 1,440 recordings**, 60 per actor, with zero generation exclusions. Every saved WAV was checked for its hash, finite decoded samples, 16-kHz rate and expected duration. Each actor has exactly one applied shift per condition. All 600 recordings from the ten actors whose shifts are within the cap have exactly identical decoded audio in the full and capped folders. Audio remains FLOAT WAV without clipping or gain matching: four full-shift and three capped files have peaks above 1, with maximum 1.0358. The separate listening exports are RMS matched and peak-scaled below full scale.

All **2,880 outputs** were remeasured using Praat 7.0.03. Every recording supplied at least ten selected output frames and some shared strong input/output frames. The full condition's estimated actor centres span **196.51–207.68 Hz**, compared with **98.89–294.05 Hz** before normalization. Its maximum residual from the 200-Hz target is **0.652 ST**. This supports successful actor-centre alignment under the audited statistic; it is not a guarantee of equal perceived voices or flat utterance pitch.

The capped output centres span **125.05–237.74 Hz**, consistent with partial normalization. The maximum residual from each actor's theoretical capped centre is 0.734 ST. Median per-recording absolute discrepancies between requested and measured median shifts are 0.0189 ST (full) and 0.0190 ST (capped). About 5% of shared strong frames differ from the requested shift by more than 1 ST; detector ambiguity, transitions, local synthesis timing and actual synthesis errors are not separated by this check.

See [measured actor centres](praat_median_experiment/measured_actor_centres.png), [actor output measurements](praat_median_experiment/output_actor_baselines.csv), [per-recording measurements](praat_median_experiment/pitch_validation.csv), and [validation summary](praat_median_experiment/pitch_validation_summary.json).

## Listening exports

RMS-matched comparisons are available for [Actor 3](praat_median_listening/actor_03_listening_sequence.wav), [Actor 5](praat_median_listening/actor_05_listening_sequence.wav) and [Actor 21](praat_median_listening/actor_21_listening_sequence.wav). Each sequence uses this group order:

1. Original.
2. Old actor mean, full shift.
3. New active Praat median, full shift.
4. Old actor mean, ±4-ST cap.
5. New active Praat median, ±4-ST cap.

Every group plays neutral, happy, angry, at normal intensity with statement/repetition 1. A single gain per condition matches the RMS across the three utterances, preserving their relative amplitudes. A shared peak scale prevents clipping. This is a listening aid, not perceptual loudness matching or a subjective quality score. Paths, gains and exact order are recorded in the [manifest](praat_median_listening/manifest.json).

## Reproduction

The cleanup retains capped audio and all generation metadata/results. Other WAVs
and one-off exploratory scripts were removed; the listening manifest records the
historical examples. See the [repository README](../../README.md) for setup,
regeneration from saved shifts, and Linux/CUDA commands.

```bash
python src/evaluate_pitch_conditions.py --device cuda --output-root output/cuda_hubert
python src/summarize_model_comparison.py --allow-incomplete --archived-audio
python src/preview_actor_normalization.py
```

The preview defaults to original versus retained capped audio. Full output pitch
validation additionally needs original diagnostic tracks and a separate current
Praat executable. The validator defaults to the retained capped condition; pass
`--conditions` explicitly to validate regenerated conditions. Historical
comparisons keep their original settings, cohorts and file hashes. Use new output
roots for changed environments, regenerated WAV headers or configuration.
