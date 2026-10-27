"""
Flask API for the Deepfake Detection backend.

Endpoints:
  GET  /health              - check the server + model are up
  POST /detect/image        - FREE: upload one image, get label + confidence
  POST /detect/video        - FREE: upload a video, get aggregated result
  POST /report              - PREMIUM: full PDF report with heatmap +
                               frequency analysis (gate this call behind
                               your RevenueCat entitlement check in Flutter)

Run locally:
  pip install -r requirements.txt
  python app.py
Then the app listens on http://localhost:5000
"""

import os
import tempfile

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

from frequency import frequency_artifact_score
from model import load_model, predict_image
from report import build_report_pdf
from video import extract_frames

app = Flask(__name__)
CORS(app)  # allow the Flutter app to call this from a different origin


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/detect/image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field name 'file'."}), 400

    file = request.files["file"]
    pil_image = Image.open(file.stream)

    result = predict_image(pil_image, with_heatmap=False)

    return jsonify(
        {
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "raw_scores": {k: round(v, 4) for k, v in result["raw_scores"].items()},
        }
    )


@app.route("/detect/video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field name 'file'."}), 400

    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        frames = extract_frames(tmp_path, max_frames=8)
        if not frames:
            return jsonify({"error": "Could not read any frames from this video."}), 400

        frame_results = [predict_image(f, with_heatmap=False) for f in frames]
        fake_votes = sum(1 for r in frame_results if r["label"].lower().startswith("deep"))
        avg_confidence = sum(r["confidence"] for r in frame_results) / len(frame_results)

        overall_label = "Deepfake" if fake_votes > len(frame_results) / 2 else "Realism"

        return jsonify(
            {
                "label": overall_label,
                "confidence": round(avg_confidence, 4),
                "frames_analyzed": len(frame_results),
                "frames_flagged_fake": fake_votes,
                "per_frame": [
                    {"label": r["label"], "confidence": round(r["confidence"], 4)}
                    for r in frame_results
                ],
            }
        )
    finally:
        os.unlink(tmp_path)


@app.route("/report", methods=["POST"])
def report():
    """
    PREMIUM endpoint. In this MVP, gating happens client-side (Flutter
    only calls this after confirming the RevenueCat entitlement is
    active). For a production app you'd also verify the purchase
    server-side via RevenueCat's REST API before generating the report —
    fine to skip for a hackathon demo.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field name 'file'."}), 400

    file = request.files["file"]
    filename = file.filename or "uploaded_file"
    pil_image = Image.open(file.stream)

    model_result = predict_image(pil_image, with_heatmap=True)
    frequency_result = frequency_artifact_score(pil_image)

    pdf_bytes = build_report_pdf(filename, model_result, frequency_result)

    from io import BytesIO

    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="deepfake_evidence_report.pdf",
    )


if __name__ == "__main__":
    load_model()  # load once at startup, not on first request
    app.run(host="0.0.0.0", port=5000, debug=True)
