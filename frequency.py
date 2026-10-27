"""
Frequency-domain analysis for deepfake detection.

GAN/diffusion-generated images tend to leave detectable artifacts in the
frequency domain (a "checkerboard" pattern from upsampling layers) that
aren't visible to the naked eye but show up clearly in a 2D FFT. This is
a real, published detection signal — cheap to compute, and most student
projects skip it entirely because they only do CNN classification.

This gives a SECOND, independent score to combine with the model's
prediction — useful for the "explainable confidence" framing the problem
statement asks for (not just one black-box number).
"""

import cv2
import numpy as np
from PIL import Image


def frequency_artifact_score(pil_image: Image.Image) -> dict:
    """
    Returns a dict with a 0-1 'artifact_score' (higher = more likely
    to contain generative-model frequency artifacts) plus the raw
    spectrum image (as a numpy array) for visualization.
    """
    gray = np.array(pil_image.convert("L"))

    # 2D FFT, shifted so low frequencies are in the center
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    # Normalize spectrum to 0-255 for visualization/reporting
    spectrum_img = magnitude_spectrum - magnitude_spectrum.min()
    spectrum_img = (spectrum_img / (spectrum_img.max() + 1e-8) * 255).astype(np.uint8)

    # Heuristic artifact score: GAN upsampling artifacts show up as
    # unusually strong energy in the high-frequency ring relative to
    # the low-frequency center. We compare mean energy in an outer
    # ring vs the center disc.
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)

    center_mask = dist_from_center < (0.15 * max_dist)
    outer_mask = dist_from_center > (0.6 * max_dist)

    center_energy = magnitude_spectrum[center_mask].mean()
    outer_energy = magnitude_spectrum[outer_mask].mean()

    # Ratio-based score, clipped to 0-1. This threshold is a starting
    # heuristic — tune it against a few real vs. fake samples before
    # relying on it for the demo.
    ratio = outer_energy / (center_energy + 1e-8)
    artifact_score = float(np.clip((ratio - 0.3) / 0.4, 0, 1))

    return {
        "artifact_score": artifact_score,
        "spectrum_image": spectrum_img,  # numpy array, grayscale
    }
