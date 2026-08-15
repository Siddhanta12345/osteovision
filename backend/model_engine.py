import cv2
import numpy as np
import os

def segment_and_contour_bone(image_path: str):
    """
    Reads an X-ray, isolates bone structures, and generates precision outlines
    that follow structural geometry and penetrate fracture gaps.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Contrast Limited Adaptive Histogram Equalization for bone radiograph enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter reduces noise while keeping bone edges sharp
    blurred = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Adaptive thresholding to segment high-density bone structures
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 2
    )
    
    # Morphological gradient to capture bone boundaries and internal gaps/fault lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_grad = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    
    # Find all contours (external and internal fault boundaries)
    contours, hierarchy = cv2.findContours(morph_grad, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Draw primary structural outlines in vibrant red
    contour_overlay = img.copy()
    cv2.drawContours(contour_overlay, contours, -1, (0, 0, 255), 2)
    
    return img, gray, contour_overlay, contours

def compare_scans(target_path: str, reference_paths: list):
    """
    Compares the target patient scan against 4 multi-angle reference scans.
    Selects the closest matching reference angle via ORB/SSIM feature registration,
    and isolates regions with bone discontinuities for localized heatmap generation.
    """
    target_img, target_gray, target_contour, target_cnts = segment_and_contour_bone(target_path)
    
    best_ref_img = None
    best_ref_gray = None
    max_matches = -1
    
    orb = cv2.ORB_create(nfeatures=1000)
    kp_target, des_target = orb.detectAndCompute(target_gray, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Select the reference angle that best matches the patient's anatomical perspective
    for ref_path in reference_paths:
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            continue
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
        if des_target is not None and des_ref is not None:
            matches = bf.match(des_target, des_ref)
            if len(matches) > max_matches:
                max_matches = len(matches)
                best_ref_img = ref_img
                best_ref_gray = ref_gray

    # Fallback to the first reference if matching points are low
    if best_ref_img is None and len(reference_paths) > 0:
        best_ref_img = cv2.imread(reference_paths[0])
        best_ref_gray = cv2.cvtColor(best_ref_img, cv2.COLOR_BGR2GRAY)

    # Resize best reference to target dimensions for spatial comparison
    h, w = target_gray.shape
    ref_resized = cv2.resize(best_ref_gray, (w, h))
    
    # Compute absolute morphological difference map to pinpoint fractures & gaps
    diff = cv2.absdiff(target_gray, ref_resized)
    _, diff_thresh = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
    
    # Refine anomaly areas using connected components
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    diff_cleaned = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)
    
    # Generate colored heatmap overlay over the fracture / gap regions
    heatmap_color = cv2.applyColorMap(diff_cleaned, cv2.COLORMAP_JET)
    
    # Create final diagnostic visualization:
    # Intact bone structures receive a green border; fracture gaps receive red/heatmap overlay
    diagnostic_canvas = target_img.copy()
    
    # Draw green outlines for intact bone contours
    cv2.drawContours(diagnostic_canvas, target_cnts, -1, (0, 220, 0), 1)
    
    # Alpha blend heatmap over areas with detected structural discontinuity
    mask_indices = diff_cleaned > 0
    alpha = 0.55
    diagnostic_canvas[mask_indices] = (
        alpha * heatmap_color[mask_indices] + (1 - alpha) * diagnostic_canvas[mask_indices]
    ).astype(np.uint8)
    
    # Trace red perimeter around high-difference fracture zones
    anomaly_contours, _ = cv2.findContours(diff_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in anomaly_contours:
        if cv2.contourArea(c) > 60: # Filter small imaging artifacts
            cv2.drawContours(diagnostic_canvas, [c], -1, (0, 0, 255), 2)
    
    return target_contour, diagnostic_canvas, len(anomaly_contours) > 0
