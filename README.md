
# README for Thermal Drone Processing Code

The code and data in this folder were used for analysis of leaf/canopy temperatures of droughted vs. nondroughted and warm-adapted vs. cool-adapted *P. fremontii* populations within the Agua Fria common garden. This is for the paper: “Remote sensing reveals inter and intraspecific variation in riparian cottonwood (*Populus* spp) response to drought” by M.M Seeley, B.C Wiebe et al.

Raw drone data were processed using FLIR Thermal Studio Pro using parameters presented in the methods section. These can be found in their original folder structure in `Ben_Thermal_and_RGB_files`. They are consolidated into one folder in `thermalFlattened_processed_40c`. The main script, `thermal_drone_processing.py`, loops through these files, applies the RGB masks to select only green leaf pixels (using different stringencies… see methods) and calculates temperature statistics from the remaining data. Statistics are calculated using both all pixels and by image to avoid issues owing to spatial autocorrelation. Functions are read in from `functions.py`.

The folder `output_sensitivity` contains statistics and plots using various masking stringencies and pixel-wise vs. image-wise calculations. The file `sensitivity_output_table.csv` contains summary statistics used in the sensitivity analysis of this paper.

Questions about the data/code can be directed to Ben Wiebe at [bcw269@nau.edu](mailto:bcw269@nau.edu).
