{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww23560\viewh12400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs28 \cf0 README for thermal drone processing code \
\
The code and data in this folder were used for analysis of leaf/canopy temperatures of droughted vs nondroughted and warm-adapted vs. cool-adapted P. fremontii populations within the Agua Fria common garden. This is for the paper: \'93Remote sensing reveals inter and intraspecific variation in riparian cottonwood (Populus spp) response to drought\'94 by M.M Seeley, B.C Wiebe et al. \
\
Raw drone data were processed using FLIR Thermal Studio Pro using parameters presented in the methods section. These can be found in their original folder structure in Ben_Thermal_and_RGB_files. They are consolidated into one folder in thermalFlattened_processed_40c. The main script, thermal_drone_processing.py, loops through these files, applies the RGB masks to select only green leaf pixels (using different stringencies\'85 see methods) and calculates temperature statistics from the remaining data. Statistics are calculated using both all pixels and by image to avoid issues owing to spatial autocorrelation. Functions are read in from functions.py. \
\
The folder \'91output_sensitivity\'92 contains statistics and plots using various masking stringencies and pixel-wise vs image-wise calculations. The file sensitivity_output_table.csv contains summaries statistics used in the sensitivity analysis of this paper.\
\
Questions about the data / code can be directed to Ben Wiebe at bcw269@nau.edu.}