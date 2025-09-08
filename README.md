# get_signers

Extract sign language from broadcast video

## What is it?
Basic video signer detection/extraction. The script will take one or more video files as input, and save cropped-and-trimmed clips in the the two dirs `signers/` and `non-signers/`.

## Usage
to detect signers in all `.mp4` files in the current directory:

`python get_signers.py *.mp4`

## How does it work?
It uses YOLO tracking and pose detection. In short:
* each video is searched for person tracks ('same' person appearing in consecutive frames in the video)
* using heuristics regarding location, size and duration of the tracks, a set of candidate tracks are extracted, and trimmed-and-cropped videos are saved for each candidate.
* to futher distinguish signers from non-signers, the amount of hand motion is measured in each candidate track. Tracks with normalized hand motion of more than 0.1 are saved to `signers`, others to `non-signers`.

## Limitations
Many! Has only been tested on a handful of files. Heuristics and thresholds are currently hard-coded in the script. etc...




### Issues found
Examples checked: one episode eatch of Curry med korre, 15 minuter på teckenspråk, Gift med fotbollen and Bästa platsen

- [x] Cropping incorrectly on the x-axis.
- [ ] Longer segments get split up into multiple shorter ones due to non-persisting person tracks of the interpreter. Worked better when trying with one higher-resolution version of the same video but roughly the same with another.
- [ ] Misclassified signer/non-signer, might need something better than a simple hand motion threshold.
    - Incorrectly classified as non-signer in short clip where mostly the hand alphabet was used.
    - Fast hand movements that have nothing to do with signing get high motion value.
- [ ] Studio dialog is cropped a bit inconsistently and the screen image is often included. How should dialog be handled?
- [ ] It says in filter_ptracks that "the signer bbox should extend to the bottom of the frame" - doesn't this only apply to interpretation?


### Resolution comparison

Detected person tracks after splitting/filtering for two different video resolutions.

| Title | 1280×720 | 1920×1080|
|-----------------:|----------------:|-----------------:|
| Curry med korre | 437/20 | 468/35 |
| 15 minuter teckenspråk | 155/75 | 154/75 |
| Gift med fotbollen | 535/28 | 585/25 |
| Bästa platsen | 272/66 | 308/69 |
