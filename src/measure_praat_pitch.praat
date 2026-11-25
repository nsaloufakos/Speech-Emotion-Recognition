# Export one filtered-autocorrelation track for generated-audio validation.
form Measure generated audio
    sentence Input_file
    sentence Output_file
endform
Read from file: input_file$
To Pitch (filtered autocorrelation): 0.01, 50, 800, 15, "no", 0.03, 0.09, 0.50, 0.055, 0.35, 0.14
Save as text file: output_file$
