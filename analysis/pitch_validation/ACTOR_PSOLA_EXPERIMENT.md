# Actor-level PSOLA normalization

> Historical experiment record. Redundant audio and exploratory demo scripts were removed during cleanup. See the [current repository instructions](../../README.md) for retained data and supported regeneration/evaluation commands.

## Motivation

Utterance normalization independently moves each estimated utterance mean toward 200 Hz. This can remove emotional pitch differences between utterances and apply quite different shifts to the same actor's voice. The proposed actor-level condition estimates one baseline per actor and applies one fixed semitone shift to all that actor's recordings. Under ideal multiplicative pitch scaling, relative pitch ratios within and between that actor's utterances are preserved.

For example, an actor baseline of 160 Hz implies a factor of 200/160 = 1.25, or about +3.86 semitones. Utterance means of 120 and 240 Hz would become 150 and 300 Hz. Their ratio remains two; neither individual utterance is forced to 200 Hz.

This rationale is consistent with the use of speaker-relative pitch measures in [cross-speaker pitch analysis](https://www.isca-archive.org/icslp_1996/maidment96_icslp.html). That reference normalizes pitch measurements for analysis; it does not establish the effectiveness of this PSOLA waveform transformation for emotion recognition.

## Baseline and conditions

The baseline is the arithmetic mean of each actor's valid original utterance-level pYIN mean F0 estimates. Each utterance receives equal weight; this is not a pooled voiced-frame mean. Estimates come from the original-audio diagnostics in `normalized_psola`, using 16-kHz pYIN with a 50–800-Hz search range. Four utterances lack three valid voiced frames and are excluded from baseline fitting; other recordings from those actors still supply their baselines. No emotion labels are used to fit a baseline. Input hashes are verified before existing estimates are reused.

Two separate conditions are generated:

- `normalized_psola_actor_mean`: one actor shift, `12 * log2(200 / actor_baseline_hz)`, applied to every recording of that actor.
- `normalized_psola_actor_mean_cap4`: the same actor shift clipped to [-4, +4] semitones. This is partial normalization: actors beyond that range are not expected to reach a 200-Hz mean. Four semitones is an exploratory listening-motivated limit, not an established artifact-free threshold.

Both use the same native-rate Praat PSOLA synthesis as the preferred listening examples, followed by resampling to 16 kHz and FLOAT WAV export. No gain matching is applied to training audio. Each condition saves actor baselines, theoretical post-shift actor means, constant applied shifts, per-file output diagnostics/hashes, exclusions, versions, and an index.

In the existing diagnostics, 1,024 of 1,436 utterance-specific shifts exceed four semitones in magnitude. Averaging by actor reduces inconsistency between an actor's recordings, but does not guarantee small shifts: 13 of 24 actor shifts still exceed four semitones. The uncapped actor shifts range from approximately -13.79 to +11.75 semitones. Input-estimator errors can still bias an actor's baseline; averaging is not an independent validation of pitch.

## Prediction setting and evaluation assumption

The user selected the current batch setting: multiple utterances for each actor are available. Baselines therefore use all available original recordings for that actor, including an unseen test actor's unlabeled batch. Actor identity groups the recordings; labels do not enter the baseline. The existing LOSO train/validation/test actor separation remains intact.

This is **batch speaker adaptation**, not a first-utterance prediction method. Test recordings influence their shared test-speaker preprocessing baseline. Comparisons with utterance-only preprocessing consequently compare both the normalization method and its access to a speaker batch. Later single-recording deployment would require a prior speaker calibration baseline or a separately evaluated online estimation strategy. No claim of that deployment capability is made here.

HuBERT evaluation uses the exact 1,429-recording cohort from the previous comparison, seeds 42/123/999, all 24 LOSO folds, and the same pretrained snapshot, final-layer time averaging, classifier, optimizer, epochs, and validation selection. The actor baselines deliberately use all original actor utterances, not only the matched evaluation cohort, reflecting the selected batch assumption. Original and utterance-level results can be compared only after confirming matching cohort/model/environment settings.

## Reproduction

```bash
conda activate speech-emotion
PYTHONPATH=/private/tmp/ser-pitch-engines python src/create_actor_psola_variants.py --workers 4
python src/evaluate_pitch_conditions.py \
  --conditions normalized_psola_actor_mean normalized_psola_actor_mean_cap4 \
  --matched-recordings output/pitch_comparison_hubert/matched_recordings.csv \
  --output-root output/pitch_comparison_actor_hubert
```

The `PYTHONPATH` prefix is unnecessary with a persistent `praat-parselmouth` installation (`requirements-pitch.txt`). A smoke run can use `--limit 4 --output-root .cache/actor-psola-smoke`; baselines still come from the complete original diagnostic set. The generator resumes verified files and rejects changed configurations. GPU feature extraction may require execution outside a sandbox that blocks Metal.

## Results (2026-11-12)

Both conditions produced all 1,440 recordings, with no Praat exclusions. Every actor has exactly one applied shift within each condition. The capped version respected [-4, +4] throughout. For the 660 recordings belonging to actors whose requested shift was already within that range, the two conditions have exactly identical decoded audio samples. WAV container hashes can differ because libsndfile's FLOAT-WAV PEAK chunk includes a timestamp; sample comparison, rather than container-byte comparison, was used for that invariant check.

Both new conditions completed all 72 seed/fold runs, with 4,287 held-out predictions each. Every seed covers the same 1,429 unique recordings exactly once, and every prediction's test actor agrees with its held-out fold. UAR/WAR and sample standard deviations were independently recomputed from those predictions. The previous comparison's cohort, model snapshot, feature extraction, package environment, splits, seeds, and training settings were checked for agreement before consolidating results.

| Audio condition | UAR (%) | WAR / accuracy (%) |
|---|---:|---:|
| Original | 58.63 ± 0.79 | 59.60 ± 0.51 |
| Historical Librosa, per utterance | 42.81 ± 1.52 | 43.92 ± 1.33 |
| PSOLA, per utterance | 55.85 ± 1.08 | 56.68 ± 0.87 |
| PSOLA, full actor-mean shift | 56.50 ± 0.77 | 57.27 ± 0.45 |
| PSOLA, actor-mean shift capped at ±4 semitones | 56.95 ± 0.37 | 58.01 ± 0.19 |

Values are mean ± sample standard deviation over three training seeds. The full actor shift gained **0.65 UAR points** over utterance PSOLA. The capped actor shift gained **1.10 points**, and remained **1.68 points below original audio**. The capped actor condition exceeded utterance PSOLA by +0.33, +1.78, and +1.18 points for seeds 42, 123, and 999; it remained below original in every seed. These are modest observed differences, not a statistical significance claim or evidence that actor-level normalization has surpassed the original-audio baseline.

The combined machine-readable results are in `output/pitch_comparison_actor_hubert/comparison_all_conditions.csv`, with seed scores and per-emotion seed recalls alongside it. Each new condition also preserves its fold metrics, histories, predictions, classifier checkpoints, and embedding cache. Input-estimator accuracy and baseline robustness remain open questions.

## Listening comparisons and shift visualization

`python src/preview_actor_normalization.py` creates RMS-matched listening sequences for actors 3, 5, and 21 in `analysis/pitch_validation/actor_normalization_listening/`. Each contains four condition groups: original, utterance PSOLA, actor PSOLA, and actor PSOLA capped at ±4. Within every group, the utterances are **neutral, happy, angry**, using normal intensity, statement 1, repetition 1. RMS matching applies one gain to the concatenated three utterances within a condition, preserving relative gains between those utterances; it is not perceptual loudness matching and is used only for listening exports. All generated sequences were checked for finite samples and peaks below full scale.

The [shift comparison figure](actor_shift_comparison.png) shows the original per-utterance shifts, the shared actor shifts, and the ±4-semitone caps. It displays requested transformations, not independently measured ground-truth output pitch.

Verification also included a four-recording smoke generation, baseline/shift formula checks, explicit cap checks, a successful zero-work resume of the smoke run, and confirmation that the existing three-condition evaluation command still reuses its completed results without changing its default cohort.
