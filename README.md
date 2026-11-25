# Speech Emotion Recognition

RAVDESS emotion classification with an MFCC BiLSTM, frozen HuBERT/wav2vec2
classifiers, HuBERT layer probes, and backbone fine-tuning implementations.
Run commands from the repository root with Python 3.11.

The retained normalized audio is
`data/audio_variants/normalized_psola_actor_praat_median_cap4`: one constant
speaker shift toward 200 Hz, capped at ±4 semitones. Speaker centres use the
median of utterance log-F0 medians from active, strong Praat pitch frames.
This uses an unlabeled batch from each speaker, including the test speaker;
it requires speaker calibration and is not single-utterance normalization.

The capped condition has the highest mean UAR in the completed matched HuBERT
and wav2vec2 comparisons. Gains over original audio are small (about 0.44 and
0.46 percentage points). MFCC comparisons remain incomplete. These are
exploratory results, not a confirmed general improvement.

## Setup on Linux

Install Git LFS before cloning. A shallow clone of the default branch avoids downloading the old ordinary-Git audio
history. `git lfs pull` retrieves the retained generated WAVs:

```bash
git lfs install
git clone --depth 1 --branch master https://github.com/nsaloufakos/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
git lfs pull
python3.11 -m venv .venv
source .venv/bin/activate
```

Install a CUDA-enabled PyTorch build using the [official installation selector](https://pytorch.org/get-started/locally/),
then install the project dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-mfcc-evaluation.txt -r requirements-pitch.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The Mac comparisons used PyTorch 2.14.1 and Transformers 5.20.1. The evaluation
requirements record the feature/analysis dependency versions. Record the actual
Linux/CUDA environment; use a fresh output directory when it differs.

For frozen encoder extraction, cache the pinned official checkpoints:

```bash
python src/cache_pretrained_model.py --repo facebook/hubert-base-ls960 --revision dba3bb02fda4248b6e082697eee756de8fe8aa8a --output .cache/hubert_model_path.txt
python src/cache_pretrained_model.py --repo facebook/wav2vec2-base --revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8 --output .cache/wav2vec_model_path.txt
```

## Continue experiments

The matched comparison cohort is fixed at 1,429 recordings in
`output/pitch_comparison_hubert/matched_recordings.csv`; the 24 actor-disjoint
LOSO folds are in `output/splits.csv`. Preserve both. Defaults compare original
audio with the retained capped condition using seeds 42, 123 and 999.

```bash
python src/evaluate_other_models.py --model mfcc --device cuda --output-root output/cuda_mfcc
python src/evaluate_other_models.py --model wav2vec --device cuda --output-root output/cuda_wav2vec
python src/evaluate_pitch_conditions.py --device cuda --output-root output/cuda_hubert
```

MFCC on CUDA uses the packed BiLSTM. Frozen encoder features are extracted on
CUDA; their small classifier heads train on CPU. Existing Mac comparison
folders remain evidence of those runs; configuration checks intentionally reject
resuming them with changed code or hardware. The interrupted MFCC histories,
predictions and interruption record remain in its original output folder.

Backbone fine-tuning and layer probing code is retained. The original model
entry points use the capped dataset by default; `SER_DATA_DIR=data/RAVDESS`
selects original audio. These entry points use the full dataset index, so they
do not automatically reproduce the matched 1,429-recording comparisons above.
Fine-tuning results and normalized layer probes are still outstanding.

## Regenerate removed audio

Other generated WAV conditions were removed. Their indices, source/output
hashes, shifts, generation settings and recognition results remain. Rebuild a
PSOLA condition directly from its saved shifts into a fresh separate root:

```bash
python src/restore_audio_variant.py --condition normalized_psola_actor_praat_median --output-root .cache/regenerated_variants
python src/evaluate_other_models.py --model mfcc --device cuda --conditions normalized_psola_actor_praat_median --variant-root .cache/regenerated_variants --output-root output/cuda_mfcc_full
```

Regeneration verifies original recording hashes and uses Parselmouth 0.4.7
(synthesis Praat 6.1.38). FLOAT WAV headers contain a creation timestamp, so a
fresh file hash can differ even when decoded samples are identical. New hashes
are recorded in the regenerated dataset; historical metadata is preserved.
Check decoded samples/measurements when comparing environments.

The original generators are retained for reproducing the older pYIN and
augmentation methods. For example, `python src/create_audio_variants.py
--conditions normalized_only --output-root .cache/legacy_variants` rebuilds the
legacy normalization in a fresh folder. `audit_pitch.py`, `summarize_pitch_audit.py`,
`create_praat_median_variants.py` and `validate_praat_median_variants.py` provide
the baseline-fitting and validation workflow. Refitting needs the separate
Praat 7.0.03 executable and diagnostic tracks. Tracks stay local and can be
recomputed; cloning does not download them. Refitting is unnecessary to rebuild
the saved speaker shifts.

## Results and submission status

- [Matched model results](analysis/model_comparison/RESULTS.md), including prediction examples and uncertainty summaries.
- [Pitch measurement study](analysis/pitch_validation/PRAAT_MEDIAN_EXPERIMENT.md) and [detector audit](analysis/pitch_validation/F0_AUDIT.md).
- [Course requirements and unfinished deliverables](analysis/project_requirements/STATUS.md). Deadline: November 27, 2026, 11:59 p.m. Toronto time.
- [Cleanup inventory](analysis/cleanup.json). All checkpoint files and remaining feature caches stay local; Git ignores them. Predictions, histories, metrics and provenance are versioned.

Completed encoder feature arrays and LaTeX scratch files were cleared. The
unfinished MFCC feature caches remain local, along with all checkpoint files.

To recompute historical tables without the deleted uncapped WAVs:

```bash
python src/summarize_model_comparison.py --allow-incomplete --archived-audio
```

This independently checks scores and held-out membership. Missing waveform hashes
are matched to saved generation diagnostics; it explicitly does not claim those
waveform bytes were rechecked. Omit `--archived-audio` for strict waveform checks.

Only the retained capped WAVs use LFS. CSV/JSON metadata uses ordinary Git;
new checkpoints, embeddings, diagnostic tracks, listening WAVs and ZIPs are
ignored. Existing Git/LFS history remains on the remote; deleting working files
does not reclaim the owner's historical storage allowance.

If LFS download allowance is exhausted, use `GIT_LFS_SKIP_SMUDGE=1 git clone ...`
and regenerate the capped condition locally with `restore_audio_variant.py`.
Point evaluation at that fresh root with `--variant-root`. This avoids LFS
downloads by recomputing the audio from the original recordings; it does not
require buying storage or changing the owner's billing settings.
