from collections import defaultdict

import cv2
import numpy as np
import sys
from ultralytics import YOLO
import subprocess
import math
from math import hypot

import matplotlib.pyplot as plt


WRIST_L = 9  # COCO order in Ultralytics pose
WRIST_R = 10


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def measure_hand_motion(
    video_path: str,
    model: object,
    conf_thres: float = 0.25,
    kpt_conf_thres: float = 0.5,
    stride: int = 1,
    normalize_by_image_diagonal: bool = True,
    verbose: bool = False,
):
    """
    Returns:
        motion_total: float, total wrist path length (pixels or normalized units)
        frames_used: int, number of processed frames
        frames_with_hands: int, frames where at least one wrist was confident
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diag = math.hypot(width, height) if normalize_by_image_diagonal else 1.0

    # model = YOLO(model_path)
    motion_total = 0.0
    frames_used = 0
    frames_with_hands = 0

    # Previous wrist positions (None until we see them)
    prev_L = None
    prev_R = None

    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        frames_used += 1

        # Run pose estimation on this frame; single image inference
        # Set verbose to False to reduce logs
        results = model.predict(frame, verbose=verbose, conf=conf_thres)

        # Choose the best person instance (if any). Ultralytics returns a list of Results (len==1 per image).
        inst_L = None
        inst_R = None

        if results and len(results) > 0:
            res = results[0]
            # res.keypoints.xy: (N,17,2) ; res.keypoints.conf: (N,17)
            # res.boxes.conf: (N,) confidence per instance
            if res.keypoints is not None and len(res.keypoints) > 0:
                # Pick the instance with max confidence (or largest box)
                # Here pick max box confidence for simplicity:
                if res.boxes is not None and len(res.boxes) > 0:
                    best_idx = int(res.boxes.conf.argmax().item())
                else:
                    best_idx = 0

                kpts_xy = res.keypoints.xy[best_idx].cpu().numpy()  # (17,2)
                kpts_cf = res.keypoints.conf[best_idx].cpu().numpy()  # (17,)

                # Left wrist
                if kpts_cf[WRIST_L] is not None and kpts_cf[WRIST_L] >= kpt_conf_thres:
                    inst_L = (float(kpts_xy[WRIST_L, 0]), float(kpts_xy[WRIST_L, 1]))
                # Right wrist
                if kpts_cf[WRIST_R] is not None and kpts_cf[WRIST_R] >= kpt_conf_thres:
                    inst_R = (float(kpts_xy[WRIST_R, 0]), float(kpts_xy[WRIST_R, 1]))

        # If at least one wrist is visible, count this frame
        if inst_L is not None or inst_R is not None:
            frames_with_hands += 1

        # Accumulate motion for each wrist separately if both current and previous exist
        if inst_L is not None and prev_L is not None:
            motion_total += euclid(inst_L, prev_L) / diag
        if inst_R is not None and prev_R is not None:
            motion_total += euclid(inst_R, prev_R) / diag

        # Update previous seen positions (only if currently visible)
        prev_L = inst_L if inst_L is not None else prev_L
        prev_R = inst_R if inst_R is not None else prev_R

        frame_idx += 1

    cap.release()

    # If hands were never visible, define movement as 0 (as requested)
    if frames_with_hands == 0:
        motion_total = 0.0

    # Normalize by number of processed frames
    avg_motion_per_frame = motion_total / frames_used if frames_used > 0 else 0.0
    return avg_motion_per_frame, motion_total, frames_used, frames_with_hands


def split_tracks_dict(
    tracks_dict,
    max_dt: float = 1.0,
    max_center_jump: float = 0.2,
    max_size_ratio: float = 1.5,
):
    """
    Split dict of tracks (id -> list of detections) on temporal/spatial discontinuities.

    Args:
        tracks_dict: dict {track_id: [{'t_sec': float, 'box': [x0,y0,x1,y1]}, ...]}
        max_dt: split if time gap > max_dt seconds
        max_center_jump: split if center distance jump > this (same units as box coords)
        max_size_ratio: split if width/height change more than this factor

    Returns:
        new_tracks: dict with new track IDs (original id + suffix) and split lists.
    """

    def center_and_wh(box):
        x0, y0, x1, y1 = box
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        w = max(1e-12, x1 - x0)
        h = max(1e-12, y1 - y0)
        return cx, cy, w, h

    new_tracks = {}
    for tid, detections in tracks_dict.items():
        if not detections:
            continue

        cur = [detections[0]]
        prev_t = detections[0]["t_sec"]
        pcx, pcy, pw, ph = center_and_wh(detections[0]["box"])
        split_idx = 0

        for det in detections[1:]:
            t = det["t_sec"]
            cx, cy, w, h = center_and_wh(det["box"])

            dt = t - prev_t
            center_jump = hypot(cx - pcx, cy - pcy)
            w_ratio = w / pw
            h_ratio = h / ph
            size_ok = (1 / max_size_ratio) <= w_ratio <= max_size_ratio and (
                1 / max_size_ratio
            ) <= h_ratio <= max_size_ratio

            need_split = (
                (dt > max_dt) or (center_jump > max_center_jump) or (not size_ok)
            )

            if need_split:
                # save current segment
                new_tracks[(tid, split_idx)] = cur
                split_idx += 1
                cur = [det]
            else:
                cur.append(det)

            prev_t, pcx, pcy, pw, ph = t, cx, cy, w, h

        if cur:
            new_tracks[(tid, split_idx)] = cur

    return new_tracks


def filter_ptracks(tracks_dict):
    """
    given a dict of person tracks, use heuristic criteria to filter out tracks that are not likely to be signers:
        - the signer bbox should extend to the bottom of the frame
        - the signer bbox should be reasonably wide and tall
        - the signer should be visible for at least X seconds

    Args:
        tracks_dict: dict {track_id: [{'t_sec': float, 'box': [x0,y0,x1,y1]}, ...]}

    Returns:
        new_tracks: filtered set of tracks
    """

    output = []

    # apply criteria
    minlen = 2
    maxbottom = 0.05
    minwidth = 0.1
    minheight = 0.5

    for trackid, frames in tracks_dict.items():
        # print('track:',trackid, frames[0]['t_sec'],' - ',frames[-1]['t_sec'])
        seglen = frames[-1]["t_sec"] - frames[0]["t_sec"]
        if seglen < minlen:
            # print('too short',seglen)
            continue

        bottom = 1 - max(d["box"][3] for d in frames)
        if bottom > maxbottom:
            # print('too high up',bottom,maxbottom)
            continue

        width = min(d["box"][2] - d["box"][0] for d in frames)
        height = min(d["box"][3] - d["box"][1] for d in frames)
        if width < minwidth or height < minheight:
            # print('too small',width,height)
            continue

        # all passed - get the overall bbox for the track
        x0 = min(d["box"][0] for d in frames)
        y0 = min(d["box"][1] for d in frames)
        x1 = max(d["box"][2] for d in frames)
        y1 = max(d["box"][3] for d in frames)
        t0 = frames[0]["t_sec"]
        t1 = frames[-1]["t_sec"]
        # print({'t0':t0,'t1':t1,'box':[x0,y0,x1,y1]})
        output.append({"t0": t0, "t1": t1, "box": [x0, y0, x1, y1]})

    return output


def get_ptracks(video_path, tracker_model, plot=False):
    print("Processing video:", video_path)
    cap = cv2.VideoCapture(video_path)

    # dict for storing person tracks
    persons = defaultdict(lambda: [])

    framenr = 0

    fwidth, fheight = None, None

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # we check every 1 second
        for xx in range(1 * fps):
            success, frame = cap.read()
            framenr += 1
        time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) * 0.001

        if success:
            fheight, fwidth = frame.shape[:2]

            # Run YOLO11 tracking on the frame, persisting tracks between frames
            result = tracker_model.track(
                frame, persist=True, tracker="botsort.yaml", verbose=False
            )[0]

            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()
                xyboxes = result.boxes.xyxyn.cpu().tolist()
                classes = result.boxes.cls

                # Visualize the result on the frame
                if plot:
                    frame = result.plot()

                # save info about persons (class = 0)
                for box, cl, track_id in zip(xyboxes, classes, track_ids):
                    if cl == 0:
                        persons[track_id].append({"t_sec": time_sec, "box": box})

            if plot:
                # Display the annotated frame
                cv2.imshow("YOLO11 Tracking", frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    return persons, fwidth, fheight


def crop_and_trim_ptrack(video_path, p, outfile, fwidth, fheight):
    x0, y0, x1, y1 = p["box"]
    # print('person:',i,'box:',x0,y0,x1,y1)
    # Calculate padding and crop dimensions
    padding = 10
    x = 2 * (fwidth * x0 - padding) // 2
    y = 2 * (fheight * y0 - padding) // 2
    w = 2 * (fwidth * (x1 - x0 + 2 * padding)) // 2
    h = 2 * (fheight * (y1 - y0 + 2 * padding)) // 2

    # Ensure the crop is within the video dimensions
    x = max(0, x)
    y = max(0, y)
    w = min(w, fwidth - x)
    h = min(h, fheight - y)

    # Use ffmpeg to crop the video
    crop_filter = f"crop=w={int(w)}:h={int(h)}:x={int(x)}:y={int(y)}"
    tpre = 1.0
    tpost = 1.0

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ss",
            str(p["t0"] - tpre),
            "-to",
            str(p["t1"] + tpost),
            "-filter:v",
            crop_filter,
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            outfile,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    # so it begins...

    tracker_model_path = "yolo11n.pt"
    pose_model_path = "yolov8n-pose.pt"

    # Load the YOLO11 model
    model = YOLO(tracker_model_path)
    pose_model = YOLO(pose_model_path)

    # Open the video file

    outdir = "."
    for video_path in sys.argv[1:]:
        plot = True
        persons, fwidth, fheight = get_ptracks(video_path, model)

        print("- detected", len(persons), "person tracks")

        persons = split_tracks_dict(persons)
        print("- after splitting: ", len(persons), "person tracks")

        persons = filter_ptracks(persons)
        print("- after filtering: ", len(persons), "person tracks")

        tmpdir = "/tmp/get_signers"
        outdir_signers = "signers"
        outdir_non_signers = "non_signers"
        subprocess.run(["mkdir", "-p", tmpdir])
        subprocess.run(["mkdir", "-p", outdir_signers])
        subprocess.run(["mkdir", "-p", outdir_non_signers])

        # Crop the video to each person track and measure hand motion

        for i, p in enumerate(persons):
            print(f"  ptrack {i}")
            outname = video_path.split("/")[-1].replace(".mp4", f"_ptrack_{i}.mp4")
            outfile = tmpdir + "/" + outname

            print("  . cropping and trimming -> ", outfile)

            crop_and_trim_ptrack(video_path, p, outfile, fwidth, fheight)

            motion, motion_total, frames_used, frames_with_hands = measure_hand_motion(
                video_path=outfile, model=pose_model, stride=30
            )

            motion_thresh = 0.1  # threshold for distinguishing signer from non-signer
            if motion < 0.1:
                print(f"  . hand motion: {motion:.2f} < {motion_thresh}")
                print(f"    -> {outdir_non_signers}")
                subprocess.run(["mv", outfile, outdir_non_signers])
            else:
                print(f"  . hand motion: {motion:.2f} >= {motion_thresh}")
                print(f"    -> {outdir_signers}")
                subprocess.run(["mv", outfile, outdir_signers])
