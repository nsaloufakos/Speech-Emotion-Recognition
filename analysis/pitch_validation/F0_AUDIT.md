# pYIN versus Praat filtered-autocorrelation audit

Run date: 2026-11-23. All 1,440 original recordings / 24 actors were measured.

## What the audit establishes

The old baseline measurements are reproducible but are unsuitable as trusted speaker pitch centres. They accept many quiet, low-probability pYIN detections and average in linear Hz, making spurious high estimates particularly influential. A different baseline can even reverse the direction of the requested shift.

The audit produces robust baseline **candidates**, detector disagreements, sensitivity measurements, and review plots. It does not establish ground-truth F0 for every frame or certify any baseline as exact. Existing audio, normalization generators, and training settings were not changed.

| Actor | Old mean baseline (Hz) | Active, strong Praat median baseline (Hz) | Old shift to 200 Hz (ST) | Candidate shift (ST) |
|---|---:|---:|---:|---:|
| 3 | 232.54 | 159.71 | -2.61 | +3.90 |
| 5 | 443.46 | 146.59 | -13.79 | +5.38 |
| 13 | 370.42 | 131.98 | -10.67 | +7.20 |
| 21 | 101.44 | 115.19 | +11.75 | +9.55 |

These differences combine **frame selection, detector choice, and a change from an arithmetic mean to a robust median**. They must not all be attributed to Praat being more accurate. The sensitivity table separates some of these effects. For Actor 5, removing quiet frames alone changes the old Hz-mean shift from -13.79 to +1.11 ST; the active pYIN median gives +3.70 ST, and the active Praat median gives +5.38 ST.

See the [actor comparison figure](f0_audit/actor_comparison.png), [baseline/coverage/quality table](f0_audit/actor_quality.csv), and [validation summary](f0_audit/validation.json).

## Methods and provenance

- Librosa 0.11.0 pYIN: 16 kHz, 50–800 Hz, 2,048-sample / 128-ms window, 512-sample / 32-ms hop. Original float32 loading and resampling reproduce the existing pipeline. The old mean-of-utterance-means shifts were reproduced to numerical precision across all 24 actors (maximum difference 8e-15 ST).
- Official Praat 7.0.03, released 2026-11-08: filtered autocorrelation, 10-ms step, 50-Hz floor, 800-Hz top, 15 candidates, standard 0.03 attenuation at top, silence threshold 0.09, voicing threshold 0.50, octave cost 0.055, jump cost 0.35, voiced/unvoiced cost 0.14. The old Parselmouth 0.4.7 dependency embeds Praat 6.1.38 and does not supply this newer algorithm.
- Primary comparisons use the same resampled waveform. Additional Praat passes use native-rate original audio and a 1,200-Hz top. Changing the top also changes Gaussian filtering and can change voicing/frame selection; it is not simply an extension of the hard F0 ceiling.
- Praat's actual frame centres are parsed from its Pitch files. Nearest frames are paired to pYIN centres only within 5 ms. No interpolation across unvoiced intervals. Different analysis-window lengths and changing pitch can still produce disagreements despite this alignment.
- Activity: 32-ms RMS windows centred at the measurement times; at least -35 dB relative to each recording's maximum RMS, with an absolute floor excluding digital silence. This is an exploratory quiet-frame mask, not a trained speech detector or manually verified voicing annotation.
- Strong Praat selection: voiced, active, selected-candidate autocorrelation strength at least 0.6. Strength is **not** a probability of correct F0. pYIN voiced probabilities are reported separately at thresholds 0.1, 0.3 and 0.5; they are also not guarantees of correct pitch.
- Candidate actor centre: median across utterance medians in semitones relative to 100 Hz, requiring at least ten selected frames per utterance. All 60 recordings per actor qualify for the Praat candidate. Each qualifying utterance has equal weight. The shift is `12 - actor_centre_ST` for the historical 200-Hz target. These are uncapped proposals, not applied transformations.
- Bootstrap intervals resample whole utterances (2,000 draws, fixed seeds), never overlapping frames. They describe sample stability, not physical F0 accuracy. Interval widths span approximately 1.43–5.77 ST, reflecting the expressive recordings and the particular baseline definition.
- No emotion labels are used for candidate baseline fitting. Emotion/neutral and repetition/statement comparisons are diagnostic only. Baselines use all recordings belonging to each actor, retaining the previous **batch speaker adaptation** protocol; they do not establish first-utterance deployment capability.

Raw/paired tracks, per-recording JSON measurements, source and track hashes, executable and script hashes, parameters, and dependency versions are retained in [f0_audit/](f0_audit/). All 1,440 track hashes passed verification.

Praat sources: [choosing a pitch method](https://www.fon.hum.uva.nl/praat/manual/how_to_choose_a_pitch_analysis_method.html), [filtered-autocorrelation settings](https://www.fon.hum.uva.nl/praat/manual/pitch_analysis_by_filtered_autocorrelation.html), [official Mac download](https://www.fon.hum.uva.nl/praat/download_mac.html). Librosa: [pYIN documentation](https://librosa.org/doc/0.11.0/generated/librosa.pyin.html).

## Validation results

### Known signals

57 engine/signal combinations: pure tones, eight-harmonic signals, the same harmonic signals with 10-dB white noise, and digital silence. Frequencies: 80, 120, 200, 320, 500 and 700 Hz. Detectors: pYIN, Praat top 800, Praat top 1200.

All clean signals satisfied median absolute error <=0.2 ST with at least 80% interior voiced coverage; all digital-silence checks remained unvoiced. One noisy stress case missed that criterion: Praat top 800 at 700 Hz had 0.248-ST median error and 81.8% coverage. The top-1200 setting passed that case. This result is retained, not silently discarded or treated as a clean-test pass. These controls validate known signals and the parser/units; they do not simulate all expressive speech or validate rapidly changing F0. See [synthetic_controls.json](f0_audit/synthetic_controls.json).

### Real-recording comparisons

- pYIN accepted 91,653 frames as voiced; **28,813 (31.4%) were outside the activity mask**. This does not prove every excluded frame was silent, but it shows substantial exposure to quiet-region estimates.
- On 48,766 shared active frames, 11,402 (23.4%) differed by more than one semitone. There were 119 octave-like differences and 17 two-octave-like differences within 0.75 ST of 12/24 ST. These are disagreements, not labelled detector errors.
- Requiring pYIN probability >=0.5 and Praat strength >=0.6 leaves only 9,044 shared active frames; 486 (5.4%) still disagree by more than one semitone. A lower disagreement rate comes with strong selection bias. For Actors 1 and 13 only one recording has ten high-probability pYIN frames, and Actor 15 has none. This threshold is unsuitable as a universal actor-baseline gate.
- An additional active agreement comparison omits that pYIN probability requirement, retains Praat strength >=0.6 and requires agreement within 0.5 ST. It covers 41–60 utterances per actor. Some resulting centres differ materially because frame/utterance coverage changes. It is a cross-check, not a replacement baseline selected after seeing the results.
- No native-rate versus 16-kHz shared voiced frame differed by more than 1 ST; maximum paired-file actor-centre change was 0.0046 ST. This audit supplies no evidence that resampling is driving the baseline problem.
- Changing Praat's top to 1200 changes actor centres by up to 0.873 ST; Actor 21 exceeds the exploratory 0.5-ST sensitivity flag. Frame inclusion can change even where common-frame F0 remains close.
- The final quality table flags Actors **5, 6, 7, 9, 11, 13, 15, 17, 20, 21 and 23** for one or more detector-centre differences, agreement-selection effects, search-setting sensitivity, or >1-ST repetition/statement differences. These flags are conservative review aids, not calibrated probabilities of correctness. Absence of a flag is not a certification.

### Acoustic inspection

36 review figures were generated: one matched normal-intensity happy recording per actor plus twelve recordings with the most shared-active disagreements. Each includes a spectrogram with both tracks, confidence/activity, a 120-ms spectrum with competing harmonic grids, and a short waveform. They deliberately select difficult local windows; **not all figures or frames have been manually adjudicated**.

Direct visual inspection included:

- [Actor 5, happy](f0_audit/review/03-01-03-01-01-01-05.png): pYIN reports values near 800 Hz in very quiet leading regions, while both tracks largely follow approximately 110–190 Hz in the speech. The selected 2.528-s window is a changing/transitional region, so its smaller detector disagreement is not assigned a definitive ground truth.
- [Actor 6, happy](f0_audit/review/03-01-03-01-01-01-06.png): a brief low Praat estimate near the onset differs sharply from pYIN. The changing spectrum/waveform does not support automatically accepting either detector at that instant.
- [Actor 10, happy](f0_audit/review/03-01-03-01-01-01-10.png): a brief Praat estimate near 77.5 Hz sits below surrounding estimates near 300 Hz. The waveform repeats much faster than a 77.5-Hz interpretation suggests; this supports a local subharmonic-tracking problem. High autocorrelation strength alone does not remove such glitches.
- [Actor 21, happy](f0_audit/review/03-01-03-01-01-01-21.png): tracks mostly agree during voiced speech, with quiet leading/trailing pYIN estimates near the search floor and a transition mismatch at the selected instant.

These observations support excluding unreliable regions and using robust summaries; they do not provide a hand-labelled reference contour.

### Existing synthesis outputs

The nine matched listening recordings (Actors 3/5/21, neutral/happy/angry, normal intensity, statement/repetition 1) were remeasured in both full and capped actor conditions: **18 outputs**, not a full output-dataset audit. Shift comparisons use original-active, nominally aligned frames with strong Praat estimates in both versions.

Across these examples, median measured shift differs from the request by less than 0.03 ST; per-file median absolute frame error is approximately 0.02–0.09 ST. This supports delivery of the requested ratio in typical selected voiced regions. Outliers remain: Actor 5's full-shift neutral example has a 4.31-ST 90th-percentile absolute frame discrepancy. Extremely lowered pitches may also fall below the 50-Hz analysis floor and disappear from shared-frame comparisons.

The shifter can therefore deliver a **badly chosen shift accurately**. For example, the old Actor 5 baseline requests -13.79 ST, leaving much of the inspected happy output near 60–80 Hz. It does not become a 200-Hz voice merely because its old pYIN baseline was 443 Hz. See [shift validation](f0_audit/existing_outputs/shift_validation.csv) and the [Actor 5 output comparison](f0_audit/existing_outputs/03-01-03-01-01-01-05.png).

## Recommended interpretation

Use the active Praat median baseline as a candidate for the next controlled listening comparison, retaining actor-specific review flags and testing output shifts. Do not use the old raw mean or an arbitrary high pYIN confidence threshold as the sole calibration statistic. Do not interpret the current table as a uniquely correct estimate of an actor's habitual pitch: emotion, phonetic content, and the chosen summary statistic affect that definition.

Changing actor centres addresses pitch height. It does not match pitch spread, formants, voice quality, or identity. Nothing here establishes improved emotion-recognition performance, validates a 200-Hz target as optimal, or warrants rewriting historical experimental results.

## Reproduction

The downloaded official executable currently lives in temporary storage and may need to be downloaded again after cleanup. The existing Conda environment was not changed. Use an installed current Praat executable with the filtered-autocorrelation command if this path disappears.

```bash
conda activate speech-emotion
python src/audit_pitch.py --praat /private/tmp/ser-praat7003.app/Contents/MacOS/Praat
python src/summarize_pitch_audit.py
python src/audit_pitch_outputs.py --praat /private/tmp/ser-praat7003.app/Contents/MacOS/Praat
```

A four-actor smoke run and a repeated hash-verified cache-resume run both completed. The Python files passed compilation. For a smoke run, add `--limit 4 --output-dir /tmp/ser-f0-smoke-new`; changed audit code, settings, dependencies or executable require a new output directory. `--skip-controls` skips only the known-signal checks on a repeat run; it does not relax source/track/config validation.
