# get-signers

Extract sign language from broadcast video

## What is it?
Basic video signer detection/extraction. The script will take one or more video files as input, and save cropped-and-trimmed clips in the the two dirs `signers/` and `non-signers/`.

## Usage
to detect signers in all `.mp4` files in the current directory:

`python get-signers.py *.mp4` 

## How does it work?
It uses YOLO tracking and pose detection. In short:
* each video is searched for person tracks ('same' person appearing in consecutive frames in the video)
* using heuristics regarding location, size and duration of the tracks, a set of candidate tracks are extracted, and trimmed-and-cropped videos are saved for each candidate.
* to futher distinguish signers from non-signers, the amount of hand motion is measured in each candidate track. Tracks with normalized hand motion of more than 0.1 are saved to `signers`, others to `non-signers`.

## Limitations
Many! Has only been tested on a handful of files. Heuristics and thresholds are currently hard-coded in the script. etc...