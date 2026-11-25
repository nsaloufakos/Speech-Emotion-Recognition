# Called by audit_pitch.py; current Praat is required (Parselmouth 0.4.7 is too old).
form Export filtered autocorrelation tracks
    sentence Input_file
    sentence Native_file
    sentence Output_prefix
endform
Read from file: input_file$
sound = selected("Sound")
To Pitch (filtered autocorrelation): 0.01, 50, 800, 15, "no", 0.03, 0.09, 0.50, 0.055, 0.35, 0.14
Save as text file: output_prefix$ + "_16k.Pitch"
Remove
selectObject: sound
To Pitch (filtered autocorrelation): 0.01, 50, 1200, 15, "no", 0.03, 0.09, 0.50, 0.055, 0.35, 0.14
Save as text file: output_prefix$ + "_top1200.Pitch"
Remove
selectObject: sound
Remove
Read from file: native_file$
To Pitch (filtered autocorrelation): 0.01, 50, 800, 15, "no", 0.03, 0.09, 0.50, 0.055, 0.35, 0.14
Save as text file: output_prefix$ + "_native.Pitch"
