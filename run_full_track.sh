#!/bin/bash
""" Full run file for vehicle tracking from simulated data.

The following job files should be reviewed before running:
- GHAME6_1.py (Simulation file)
- process_video.sh
- image_data_noise.py 
- track_job.py
"""

python3 -m GHAME6_1.py # Run simulation

python3 -m idmoc.vehicle_master.vehicle_image_generation --file=GHAME6_1_results.csv --save
# (Close video files)

mv east.avi video/
mv above.avi video/
mv north.avi video/

cd particle-tracking
./process_video.sh
cd ../

mv video/east/tracking_data.txt ./tracking_dataE.txt
mv video/above/tracking_data.txt ./tracking_dataA.txt
mv video/north/tracking_data.txt ./tracking_dataN.txt

./image_data_noise.py

# python3 -m idmoc.motiontracker.vehicle_track --job=track_job.py
python3 -m idmoc.motiontracker.vehicle_track --job=track_job.py --plot-blob
# python3 -m idmoc.motiontracker.vehicle_track --job=track_job.py --sim-error
# python3 -m idmoc.motiontracker.vehicle_track --job=track_job.py --sim-error --plot-blob



