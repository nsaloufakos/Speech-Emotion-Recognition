"""Speech pitch shifting shared by listening tests and dataset generation."""

import numpy as np


def psola_pitch_shift(audio, sample_rate, semitones, fmin=50., fmax=800.):
    import parselmouth

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("PSOLA requires a finite, nonempty mono waveform")
    if not np.isfinite(semitones):
        raise ValueError("Pitch shift must be finite")
    call = parselmouth.praat.call
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    manipulation = call(sound, "To Manipulation", .01, fmin, fmax)
    tier = call(manipulation, "Extract pitch tier")
    if call(tier, "Get number of points") == 0:
        raise ValueError("Praat found no voiced pitch points")
    call(tier, "Multiply frequencies", sound.xmin, sound.xmax, float(2 ** (semitones / 12)))
    call([tier, manipulation], "Replace pitch tier")
    shifted = call(manipulation, "Get resynthesis (overlap-add)").values[0].copy()
    if shifted.shape != audio.shape or not np.isfinite(shifted).all():
        raise ValueError("PSOLA returned invalid samples or changed the sample count")
    return shifted
