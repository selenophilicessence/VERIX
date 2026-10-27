# Deepfake Detection Backend

A Flask API that classifies images/videos as real or deepfake using a
pretrained Vision Transformer model (no training required), plus a
frequency-domain artifact check and a PDF evidence report.

## What's in here
- `app.py` — the Flask server, all routes
- `model.py` — loads the pretrained ViT model, runs predictions,
  builds attention-based heatmaps
- `frequency.py` — FFT-based artifact detection (the differentiator —
  most student projects only do CNN classification)
- `video.py` — pulls sample frames out of an uploaded video
- `report.py` — builds the PDF evidence report (premium feature)

## Setup (Windows)
1. Make sure Python 3.10+ is installed (`python --version` in a terminal).
2. Open a terminal in this `deepfake_backend` folder.
3. Create a virtual environment (keeps these packages separate from
   anything else on your machine):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   You'll know it worked if your terminal prompt now starts with `(venv)`.
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   This will take a few minutes — it's downloading PyTorch, which is large.
5. Run the server:
   ```
   python app.py
   ```
   First run will also download the pretrained model (~350MB) from
   Hugging Face — that's normal, only happens once.
6. You should see `Model loaded.` then `Running on http://0.0.0.0:5000`.

## Testing it works (before connecting Flutter)
With the server running, open a **second** terminal and try:
```
curl -X POST -F "file=@path\to\some\image.jpg" http://localhost:5000/detect/image
```
You should get back JSON like:
```json
{"label": "Realism", "confidence": 0.94, "raw_scores": {...}}
```

## Connecting from the Flutter app
Your Android emulator can't reach `localhost` directly — it needs
`http://10.0.2.2:5000` instead (that's a special address that means
"my computer" from inside the emulator). A real physical phone on the
same WiFi would instead use your computer's local IP (e.g.
`http://192.168.1.x:5000`).

## Where the RevenueCat gating happens
This backend doesn't know anything about purchases — that's intentional.
The Flutter app checks the RevenueCat entitlement first, and only calls
`/detect/image` or `/detect/video` for the free tier, or `/report` after
confirming the user is premium.

## Next step
Once this is running and you've tested it with `curl` or Postman, tell
me and we'll build the Flutter upload screen that talks to it.
