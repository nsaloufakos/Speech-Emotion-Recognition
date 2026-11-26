# Matched P20 model comparison

All completed cells use the same 1,429 recordings, 24 LOSO folds and seeds 42, 123 and 999. Values below are mean +/- sample standard deviation in percent. Original audio uses no speaker calibration; normalized conditions use each speaker's available batch.

| Model | Audio condition | UAR (%) | WAR (%) |
|---|---|---:|---:|
| mfcc | Original | 35.65 +/- 2.18 | 37.79 +/- 2.07 |
| mfcc | New full | 36.02 +/- 0.53 | 38.35 +/- 0.50 |
| mfcc | New cap 4 | 36.31 +/- 2.87 | 38.63 +/- 3.00 |
| wav2vec | Original | 43.62 +/- 0.37 | 44.65 +/- 0.39 |
| wav2vec | New full | 44.05 +/- 1.06 | 44.90 +/- 0.99 |
| wav2vec | New cap 4 | 44.09 +/- 0.38 | 45.18 +/- 0.32 |
| hubert | Original | 58.63 +/- 0.79 | 59.60 +/- 0.51 |
| hubert | New full | 58.54 +/- 0.62 | 59.32 +/- 0.39 |
| hubert | New cap 4 | 59.06 +/- 0.62 | 59.79 +/- 0.63 |

Prediction examples are concrete seed-42 classification changes, not causal diagnoses. Confusion figures summarize the same recordings across seeds, not 4,287 independent examples. Hardware/execution differs across model families but is fixed within each model's original/full/capped comparison.

actor_resampling_intervals.json resamples the 24 test actors jointly across conditions and averages over the three fixed seed predictions (10,000 draws). These are conditional uncertainty summaries; they do not account for retraining, overlapping LOSO training sets, or exploratory method selection, and are not confirmatory significance tests.

Data: main_results.csv, seed_scores.csv, per_emotion_recall.csv, per_actor_scores.csv, paired_comparisons.json, prediction_examples.csv and results_validation.json.
