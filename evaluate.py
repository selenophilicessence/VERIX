"""
Measure accuracy on YOUR images so you know what to trust.

Folder layout (next to this file):
  testset/real/   <- 10-20 genuine photos (selfies, friends, internet)
  testset/fake/   <- 10-20 deepfakes / AI-generated faces

Run:   python evaluate.py

Prints each model's accuracy, the ensemble's accuracy, and the best
cutoff. If one model is bad, delete it from MODEL_NAMES in model.py.
Free sources for fakes: thispersondoesnotexist.com screenshots, Kaggle
"140k real and fake faces", or deepfake demo videos (screenshot frames).
"""

import os
import numpy as np
from PIL import Image

import model as m

EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_folder(path):
    if not os.path.isdir(path):
        return []
    return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(EXTS)]


def main():
    m.load_model()
    rows = []  # (is_fake, {model: p}, ensemble_p, filename)
    for label, folder in ((0, "testset/real"), (1, "testset/fake")):
        for path in load_folder(folder):
            r = m.predict_image(Image.open(path))
            rows.append((label, r["per_model"], r["raw_scores"]["Deepfake"], path))

    if not rows:
        print("No images found. Put photos in testset/real and testset/fake.")
        return

    y = np.array([r[0] for r in rows])
    print(f"\n{len(rows)} images ({int((y == 0).sum())} real, {int((y == 1).sum())} fake)\n")

    names = list(rows[0][1].keys())
    for n in names:
        p = np.array([r[1][n] for r in rows])
        acc = ((p > 0.5).astype(int) == y).mean()
        print(f"  {acc * 100:5.1f}%  {n}")

    ens = np.array([r[2] for r in rows])
    print(f"  {((ens > 0.5).astype(int) == y).mean() * 100:5.1f}%  ENSEMBLE (cutoff 0.5)")

    best_t, best_acc = 0.5, 0
    for t in np.arange(0.1, 0.95, 0.05):
        acc = ((ens > t).astype(int) == y).mean()
        if acc > best_acc:
            best_t, best_acc = t, acc
    print(f"\nBest ensemble cutoff: {best_t:.2f} -> {best_acc * 100:.1f}% accuracy")
    print("Set REAL_BELOW / FAKE_ABOVE in model.py around that value.\n")

    print("Wrong predictions (at 0.5):")
    for lab, _, p, path in rows:
        if int(p > 0.5) != lab:
            print(f"  {'REAL' if lab == 0 else 'FAKE'} scored {p:.2f}: {path}")


if __name__ == "__main__":
    main()