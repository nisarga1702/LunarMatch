import numpy as np
import cv2

def compute_sun_direction(azimuth_deg, elevation_deg):
    """Convert sun azimuth and elevation to 3D direction vector [x, y, z].
    Returns: np.ndarray (3,)"""
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)
    return np.array([
        np.cos(el_rad) * np.sin(az_rad),
        np.cos(el_rad) * np.cos(az_rad),
        np.sin(el_rad)
    ])

def compute_illumination_map(image_gray, sun_azimuth_deg, sun_elevation_deg):
    """Estimate expected illumination pattern given sun geometry.
    Uses image gradient to approximate surface normals.
    Returns: illumination_map (H,W) float in [0,1]"""
    # Assuming image_gray reflects some topography, rough normal estimation
    blur = cv2.GaussianBlur(image_gray, (5, 5), 0).astype(np.float32)
    dzdx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    
    # Scale normals roughly
    z_scale = 50.0
    h, w = image_gray.shape
    normal = np.dstack((-dzdx, -dzdy, np.ones((h, w), dtype=np.float32) * z_scale))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= (norm + 1e-8)
    
    sun_dir = compute_sun_direction(sun_azimuth_deg, sun_elevation_deg)
    
    intensity = np.sum(normal * sun_dir, axis=2)
    intensity = np.clip(intensity, 0, 1)
    return intensity

def compute_shadow_map(image_gray, sun_azimuth_deg, sun_elevation_deg, threshold=0.3):
    """Predict approximate shadow regions based on sun angle and image features.
    Returns: shadow_mask (H,W) bool, shadow_direction_deg float"""
    # Simply thresholding the estimated illumination map for demo
    illum_map = compute_illumination_map(image_gray, sun_azimuth_deg, sun_elevation_deg)
    shadow_mask = illum_map < threshold
    return shadow_mask, sun_azimuth_deg

def illumination_consistency_score(image_a, image_b, sun_az_a, sun_el_a, sun_az_b, sun_el_b):
    """Score how consistent the illumination patterns are between two images
    given their respective sun geometries. Higher = more consistent.
    Returns: float in [0, 1]"""
    # In a real scenario, this would inverse-render or compare expected shadow changes.
    # Here we do a simple heuristic comparing image intensities.
    mean_a = np.mean(image_a) / 255.0
    mean_b = np.mean(image_b) / 255.0
    
    # Very basic placeholder metric
    diff = abs(mean_a - mean_b)
    score = np.clip(1.0 - diff, 0, 1)
    return score

def visualize_sun_direction(image, sun_azimuth_deg, sun_elevation_deg):
    """Draw sun direction arrows overlaid on image. Returns annotated BGR image."""
    res = image.copy()
    h, w = image.shape[:2]
    center = (w - 50, 50)
    length = 30
    
    az_rad = np.radians(sun_azimuth_deg)
    end_pt = (
        int(center[0] + length * np.sin(az_rad)),
        int(center[1] - length * np.cos(az_rad)) # y is flipped in image coords
    )
    
    cv2.arrowedLine(res, center, end_pt, (0, 255, 255), 2, tipLength=0.3)
    cv2.putText(res, f"Sun: {sun_azimuth_deg} deg", (w - 120, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return res

def visualize_shadow_analysis(image, shadow_mask, sun_azimuth_deg):
    """Visualize shadow regions with sun direction context. Returns annotated BGR image."""
    res = image.copy()
    # Apply blue tint to shadows
    shadow_overlay = np.zeros_like(res)
    shadow_overlay[shadow_mask] = [255, 0, 0] # Blue in BGR
    
    cv2.addWeighted(shadow_overlay, 0.5, res, 1.0, 0, res)
    return visualize_sun_direction(res, sun_azimuth_deg, 35)

def compute_confidence_breakdown(structure_sim, geometry_consistency, illumination_consistency):
    """Compute the combined Correspondence Confidence Score.
    Returns: dict with 'structure', 'geometry', 'illumination', 'combined' scores"""
    combined = 0.5 * structure_sim + 0.3 * geometry_consistency + 0.2 * illumination_consistency
    return {
        'structure': structure_sim,
        'geometry': geometry_consistency,
        'illumination': illumination_consistency,
        'combined': combined
    }
