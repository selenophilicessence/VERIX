"""Extracts evenly-spaced frames from an uploaded video for analysis."""

import cv2
from PIL import Image


def extract_frames(video_path: str, max_frames: int = 8):
    """
    Pulls up to `max_frames` evenly-spaced frames from a video file.
    Returns a list of PIL Images. Sampling frames instead of every single
    one keeps this fast enough for a hackathon demo.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    step = max(total_frames // max_frames, 1)
    frames = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame_bgr = cap.read()
        if not success:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames
