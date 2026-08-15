import os
import shutil
import uuid
import cv2
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.model_engine import segment_and_contour_bone, compare_scans

app = FastAPI(title="Comparative Bone Radiograph Diagnostic Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/processed", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

@app.post("/api/v1/analyze-comparative")
async def analyze_comparative(
    target_scan: UploadFile = File(...),
    reference_angles: List[UploadFile] = File(...)
):
    if len(reference_angles) < 2:
        raise HTTPException(
            status_code=400, 
            detail=f"Please provide at least 2 reference angles (up to 6 supported). Received {len(reference_angles)}."
        )

    unique_id = str(uuid.uuid4())
    
    # Save Target Patient Scan
    target_ext = target_scan.filename.split(".")[-1].lower()
    target_filename = f"target_{unique_id}.{target_ext}"
    target_path = os.path.join("static", "uploads", target_filename)
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(target_scan.file, buffer)
        
    # Save up to 6 Reference Angle Scans
    ref_paths = []
    for idx, ref in enumerate(reference_angles):
        ref_ext = ref.filename.split(".")[-1].lower()
        ref_filename = f"ref_{idx+1}_{unique_id}.{ref_ext}"
        ref_path = os.path.join("static", "uploads", ref_filename)
        with open(ref_path, "wb") as buffer:
            shutil.copyfileobj(ref.file, buffer)
        ref_paths.append(ref_path)

    # Run Computer Vision Pipeline
    try:
        contour_img, diagnostic_img, has_fracture = compare_scans(target_path, ref_paths)
        
        contour_filename = f"contour_{unique_id}.jpg"
        diag_filename = f"diag_{unique_id}.jpg"
        
        contour_out_path = os.path.join("static", "processed", contour_filename)
        diag_out_path = os.path.join("static", "processed", diag_filename)
        
        cv2.imwrite(contour_out_path, contour_img)
        cv2.imwrite(diag_out_path, diagnostic_img)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision Engine Error: {str(e)}")

    return {
        "status": "success",
        "has_fracture_detected": has_fracture,
        "target_contour_url": f"/static/processed/{contour_filename}",
        "diagnostic_comparison_url": f"/static/processed/{diag_filename}",
        "original_target_url": f"/static/uploads/{target_filename}"
    }
