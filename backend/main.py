import os
import io
import uuid
import cv2
import numpy as np
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="OsteoVision API", version="2.0")

# Allow CORS for GitHub Pages and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PROCESSED_DIR = os.path.join(STATIC_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def health_check():
    return {"status": "online", "platform": "OsteoVision AI"}

@app.post("/api/v1/analyze-comparative")
async def analyze_comparative(
    target_scan: UploadFile = File(...),
    reference_angles: List[UploadFile] = File(...)
):
    try:
        # Read target scan
        target_bytes = await target_scan.read()
        nparr_target = np.frombuffer(target_bytes, np.uint8)
        target_img = cv2.imdecode(nparr_target, cv2.IMREAD_COLOR)

        if target_img is None:
            raise HTTPException(status_code=400, detail="Invalid target image file.")

        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.equalizeHist(target_gray)

        # 1. Generate structural bone contours
        blurred = cv2.GaussianBlur(target_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_overlay = target_img.copy()
        cv2.drawContours(contour_overlay, contours, -1, (0, 0, 255), 2)

        # 2. Multi-angle baseline alignment & difference computation
        accumulated_diff = np.zeros_like(target_gray, dtype=np.float32)
        valid_refs = 0

        for ref_file in reference_angles:
            ref_bytes = await ref_file.read()
            nparr_ref = np.frombuffer(ref_bytes, np.uint8)
            ref_img = cv2.imdecode(nparr_ref, cv2.IMREAD_GRAYSCALE)

            if ref_img is not None:
                ref_resized = cv2.resize(ref_img, (target_gray.shape[1], target_gray.shape[0]))
                ref_equalized = cv2.equalizeHist(ref_resized)
                diff = cv2.absdiff(target_gray, ref_equalized)
                accumulated_diff += diff.astype(np.float32)
                valid_refs += 1

        if valid_refs > 0:
            avg_diff = accumulated_diff / valid_refs
        else:
            avg_diff = cv2.absdiff(target_gray, cv2.GaussianBlur(target_gray, (21, 21), 0)).astype(np.float32)

        norm_diff = cv2.normalize(avg_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(norm_diff, 85, 255, cv2.THRESH_BINARY)
        
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        diagnostic_overlay = cv2.addWeighted(target_img, 0.65, heatmap, 0.35, 0)

        # Mark localized discontinuities
        discontinuity_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        has_fracture = False

        for c in discontinuity_contours:
            if cv2.contourArea(c) > 120:
                has_fracture = True
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(diagnostic_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(diagnostic_overlay, "GAP DETECTED", (x, max(y - 6, 12)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Save result artifacts
        session_id = uuid.uuid4().hex[:8]
        orig_filename = f"orig_{session_id}.jpg"
        contour_filename = f"contour_{session_id}.jpg"
        diag_filename = f"diag_{session_id}.jpg"

        cv2.imwrite(os.path.join(PROCESSED_DIR, orig_filename), target_img)
        cv2.imwrite(os.path.join(PROCESSED_DIR, contour_filename), contour_overlay)
        cv2.imwrite(os.path.join(PROCESSED_DIR, diag_filename), diagnostic_overlay)

        return {
            "has_fracture_detected": bool(has_fracture),
            "original_target_url": f"https://osteovision-api.onrender.com/static/processed/{orig_filename}",
            "target_contour_url": f"https://osteovision-api.onrender.com/static/processed/{contour_filename}",
            "diagnostic_comparison_url": f"https://osteovision-api.onrender.com/static/processed/{diag_filename}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
