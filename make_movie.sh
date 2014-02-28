#!/bin/bash
echo "If you go to NYU film school you get to make cool movies like this."
cd ./movie_frames
rm *.png
cd ..
python generate_images.py
ffmpeg -i ./movie_frames/frame_%d.png -loop 5 -pix_fmt yuv420p particle_motion.mp4
open particle_motion.mp4
