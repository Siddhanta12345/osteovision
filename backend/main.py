import os
import io
import base64
import cv2
import numpy as np
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="OsteoVision API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(img_bgr) -> str:
    success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise ValueError("Failed to encode image to JPEG.")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

@app.get("/")
def health_check():
    return {"status": "online", "platform": "OsteoVision AI"}

@app.post("/api/v1/analyze-comparative")
async def analyze_comparative(
    target_scan: UploadFile = File(...),
    reference_angles: List[UploadFile] = File(...)
):
    try:
        target_bytes = await target_scan.read()
        if not target_bytes:
            raise HTTPException(status_code=400, detail="Target scan file is empty.")

        nparr_target = np.frombuffer(target_bytes, np.uint8)
        target_img = cv2.imdecode(nparr_target, cv2.IMREAD_COLOR)

        if target_img is None:
            raise HTTPException(status_code=400, detail="Unable to decode target image.")

        target_img = cv2.resize(target_img, (600, 600))
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.equalizeHist(target_gray)

        blurred = cv2.GaussianBlur(target_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 130)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contour_overlay = target_img.copy()
        cv2.drawContours(contour_overlay, contours, -1, (0, 0, 255), 2)

        accumulated_diff = np.zeros_like(target_gray, dtype=np.float32)
        valid_refs = 0

        for ref_file in reference_angles:
            ref_bytes = await ref_file.read()
            if not ref_bytes:
                continue
            nparr_ref = np.frombuffer(ref_bytes, np.uint8)
            ref_img = cv2.imdecode(nparr_ref, cv2.IMREAD_GRAYSCALE)

            if ref_img is not None:
                ref_resized = cv2.resize(ref_img, (target_gray.shape[1], target_gray.shape[0]))
                ref_equalized = cv2.equalizeHist(ref_resized)
                diff = cv2.absdiff(target_gray, ref_equalized)
                accumulated_diff += diff.astype(np.float32)
                valid_refs += 1

        if valid_refs > 0:
            avg_diff = accumulated_diff / float(valid_refs)
        else:
            avg_diff = cv2.absdiff(target_gray, cv2.GaussianBlur(target_gray, (21, 21), 0)).astype(np.float32)

        norm_diff = cv2.normalize(avg_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(norm_diff, 80, 255, cv2.THRESH_BINARY)

        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        diagnostic_overlay = cv2.addWeighted(target_img, 0.65, heatmap, 0.35, 0)

        discontinuity_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_fracture = False

        for c in discontinuity_contours:
            if cv2.contourArea(c) > 100:
                has_fracture = True
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(diagnostic_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(diagnostic_overlay, "GAP DETECTED", (x, max(y - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        return {
            "has_fracture_detected": bool(has_fracture),
            "original_target_url": image_to_base64(target_img),
            "target_contour_url": image_to_base64(contour_overlay),
            "diagnostic_comparison_url": image_to_base64(diagnostic_overlay),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {str(e)}")
