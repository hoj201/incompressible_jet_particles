#!/bin/sh

rm -f ./movie_frames/*.png
python generate_images.py

if [ `uname` = 'Linux' ]; then
	avconv -y -i ./movie_frames/frame_%03d.png  -pix_fmt yuv420p particle_motion.mp4
else
	ffmpeg -i ./movie_frames/frame_%03d.png -loop 5 -pix_fmt yuv420p particle_motion.mp4
	open particle_motion.mp4
fi
