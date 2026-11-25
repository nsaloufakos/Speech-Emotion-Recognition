"""Add coverage, estimator sensitivity and baseline stability to an F0 audit."""

import argparse
import json
from pathlib import Path

from audit_pitch import ROOT, masks, semitones, sha
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "analysis/pitch_validation/f0_audit")
    args = parser.parse_args()
    output = args.audit_dir.resolve()
    records = [json.loads(p.read_text()) for p in sorted((output / "records").glob("*.json"))]
    rows = []
    for record in records:
        tr = dict(np.load(output / "tracks" / f"{record['file']}.npz"))
        if sha(output / "tracks" / f"{record['file']}.npz") != record["track_sha256"]:
            raise ValueError("Track checksum failed")
        selected = masks(tr)
        row = {k: record[k] for k in ("file", "actor_id", "emotion", "repetition", "statement")}
        for method, f0, mask in (
            ("raw_hz_mean", tr["pyin_f0"], selected["pyin_raw"]),
            ("active_hz_mean", tr["pyin_f0"], selected["pyin_active"]),
            ("raw_log_mean", tr["pyin_f0"], selected["pyin_raw"]),
            ("active_log_mean", tr["pyin_f0"], selected["pyin_active"]),
            ("native_praat", tr["praat_native"], tr["active"] & (tr["strength_native"] >= .6) & np.isfinite(tr["praat_native"])),
            ("top1200_praat", tr["praat_top1200"], tr["active"] & (tr["strength_top1200"] >= .6) & np.isfinite(tr["praat_top1200"])),
            ("active_agreement", tr["praat_16k"], selected["shared_active"] & (tr["strength_16k"] >= .6)
             & (np.abs(semitones(tr["pyin_f0"]) - semitones(tr["praat_16k"])) <= .5)),
        ):
            if mask.sum() >= (3 if method == "raw_hz_mean" else 10):
                if "hz_mean" in method:
                    row[method] = float(f0[mask].mean())
                elif "log_mean" in method:
                    row[method] = float(semitones(f0[mask]).mean())
                else:
                    row[method] = float(np.median(semitones(f0[mask])))
        m = record["measurements"]["praat_strong"]
        if m["frames"] >= 10:
            row["praat_strong"] = m["median_st"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    actors = pd.read_csv(output / "actor_baselines.csv").set_index("actor_id")
    for actor, group in frame.groupby("actor_id"):
        for method in ("raw_hz_mean", "active_hz_mean", "raw_log_mean", "active_log_mean", "native_praat", "top1200_praat", "active_agreement"):
            values = group[method].dropna()
            centre = values.mean() if "mean" in method else values.median()
            if "hz_mean" in method:
                centre = 12 * np.log2(centre / 100)
            actors.loc[actor, f"{method}_shift_st"] = 12 - centre
            if method == "active_agreement":
                actors.loc[actor, "active_agreement_recordings"] = len(values)
                actors.loc[actor, "active_agreement_baseline_hz"] = 100 * 2 ** (centre / 12)
        for method, label in (("native_praat", "native_rate"), ("top1200_praat", "pitch_top")):
            # Paired files so changing coverage cannot masquerade as a centre shift.
            matched = group.dropna(subset=[method, "praat_strong"])
            actors.loc[actor, f"{label}_paired_recordings"] = len(matched)
            actors.loc[actor, f"{label}_baseline_delta_st"] = matched[method].median() - matched.praat_strong.median()
        repetitions = group.groupby("repetition").praat_strong.median()
        actors.loc[actor, "repetition_baseline_gap_st"] = repetitions.max() - repetitions.min()
        statements = group.groupby("statement").praat_strong.median()
        actors.loc[actor, "statement_baseline_gap_st"] = statements.max() - statements.min()
        neutral = group[group.emotion == "neutral"].praat_strong.dropna()
        actors.loc[actor, "neutral_baseline_hz_diagnostic_only"] = 100 * 2 ** (neutral.median() / 12)
        # Broad original coverage and convergence checks are review flags, not
        # empirically calibrated probabilities of correctness.
        row = actors.loc[actor]
        flags = []
        if row.praat_strong_recordings < .8 * row.recordings:
            flags.append("limited_praat_coverage")
        if abs(row.native_rate_baseline_delta_st) > .5:
            flags.append("native_rate_sensitive")
        if abs(row.pitch_top_baseline_delta_st) > .5:
            flags.append("pitch_top_sensitive")
        if row.repetition_baseline_gap_st > 1:
            flags.append("repetition_sensitive")
        if row.statement_baseline_gap_st > 1:
            flags.append("statement_sensitive")
        if abs(row.pyin_active_shift_st - row.praat_strong_shift_st) > 1:
            flags.append("detectors_differ_on_actor_centre")
        if abs(row.active_agreement_shift_st - row.praat_strong_shift_st) > 1:
            flags.append("agreement_selection_sensitive")
        actors.loc[actor, "review_flags"] = ";".join(flags) if flags else "no_flag_at_exploratory_thresholds"
    actors.reset_index().to_csv(output / "actor_quality.csv", index=False)
    frame.to_csv(output / "baseline_sensitivity_recordings.csv", index=False)
    checks = {
        "summary_code_sha256": sha(__file__), "audit_config_sha256": sha(output / "config.json"),
        "recordings": len(records), "actors": len(actors),
        "all_track_hashes_valid": True,
        "old_hz_mean_max_reproduction_error_st": float(np.abs(actors.raw_hz_mean_shift_st - actors.old_shift_st).max()),
        "native_rate_max_actor_delta_st": float(actors.native_rate_baseline_delta_st.abs().max()),
        "pitch_top_max_actor_delta_st": float(actors.pitch_top_baseline_delta_st.abs().max()),
        "actors_without_exploratory_flags": int((actors.review_flags == "no_flag_at_exploratory_thresholds").sum()),
        "limits": "Bootstrap intervals describe utterance-sampling stability, not F0 accuracy. Consensus filtering changes frame and utterance coverage. No manual ground-truth contour exists. Neutral and emotion fields are diagnostics, not inputs to proposed baselines."
    }
    (output / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    print(actors[["old_baseline_hz", "praat_strong_baseline_hz", "old_shift_st", "praat_strong_shift_st", "review_flags"]].round(2).to_string())


if __name__ == "__main__":
    main()
