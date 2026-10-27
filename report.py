"""
Generates a structured PDF report: methodology, per-frame/image scores,
and the frequency-artifact score. This is the part that ties the tool to
the actual government use case (evidence documentation for courts).
"""

import io
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def build_report_pdf(filename: str, model_result: dict, frequency_result: dict) -> bytes:
    """
    Returns the PDF as raw bytes (write directly to a Flask response,
    no temp file needed).
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 25 * mm

    def line(text, size=11, bold=False, gap=7 * mm):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(20 * mm, y, text)
        y -= gap

    line("Digital Evidence Verification Report", size=16, bold=True, gap=10 * mm)
    line(f"File analyzed: {filename}")
    line(f"Generated: {datetime.now(timezone.utc).isoformat()} UTC")
    line("Tool: Deepfake Detector (ViT model + frequency-domain analysis)")
    y -= 5 * mm

    line("Classification Result", size=13, bold=True)
    line(f"Verdict: {model_result['label']}")
    line(f"Model confidence: {model_result['confidence'] * 100:.1f}%")
    for label, score in model_result["raw_scores"].items():
        line(f"    {label}: {score * 100:.1f}%", size=10)
    y -= 5 * mm

    line("Frequency-Domain Analysis", size=13, bold=True)
    line(f"Artifact score: {frequency_result['artifact_score']:.2f} (0 = clean, 1 = strong artifacts)")
    line("Method: 2D FFT comparing high-frequency ring energy to low-frequency")
    line("center energy. GAN/diffusion upsampling layers leave detectable")
    line("high-frequency artifacts invisible to the naked eye.")
    y -= 5 * mm

    line("Methodology Notes", size=13, bold=True)
    line("1. Frame/image classified using a ViT model fine-tuned on real vs.")
    line("   deepfake face datasets.")
    line("2. Independent frequency-domain check for generative-model artifacts.")
    line("3. Attention-based heatmap highlights regions the model focused on")
    line("   (see attached image).")
    y -= 5 * mm

    line(
        "This report is generated for investigative reference. It is not a",
        size=9,
    )
    line(
        "substitute for forensic expert testimony in legal proceedings.",
        size=9,
    )

    # Attach the frequency spectrum image if there's room, else new page
    if "spectrum_image" in frequency_result:
        if y < 90 * mm:
            c.showPage()
            y = height - 25 * mm
        from PIL import Image
        import numpy as np

        spectrum_pil = Image.fromarray(frequency_result["spectrum_image"]).convert("L")
        img_reader = ImageReader(spectrum_pil)
        c.drawString(20 * mm, y, "Frequency spectrum (for reference):")
        y -= 5 * mm
        c.drawImage(img_reader, 20 * mm, y - 60 * mm, width=60 * mm, height=60 * mm)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
