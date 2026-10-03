"""Pre-build a catalog of 5 famous lunar craters with synthetic reference images.
Called once on first startup if catalog is empty."""

import cv2
import numpy as np

FAMOUS_CRATERS = [
    {
        "name": "Tycho",
        "label": "C-001",
        "latitude": -43.31,
        "longitude": -11.36,
        "diameter_km": 85.0,
        "description": "Prominent young crater with bright ray system",
        "sun_azimuth": 135.0,
        "sun_elevation": 30.0,
    },
    {
        "name": "Copernicus",
        "label": "C-002",
        "latitude": 9.62,
        "longitude": -20.08,
        "diameter_km": 93.0,
        "description": "Large complex crater with terraced walls",
        "sun_azimuth": 90.0,
        "sun_elevation": 40.0,
    },
    {
        "name": "Aristarchus",
        "label": "C-003",
        "latitude": 23.73,
        "longitude": -47.49,
        "diameter_km": 40.0,
        "description": "Brightest large crater on the Moon",
        "sun_azimuth": 160.0,
        "sun_elevation": 25.0,
    },
    {
        "name": "Kepler",
        "label": "C-004",
        "latitude": 8.12,
        "longitude": -38.01,
        "diameter_km": 31.0,
        "description": "Young impact crater with bright rays",
        "sun_azimuth": 110.0,
        "sun_elevation": 35.0,
    },
    {
        "name": "Plato",
        "label": "C-005",
        "latitude": 51.62,
        "longitude": -9.38,
        "diameter_km": 101.0,
        "description": "Large dark-floored crater near Mare Imbrium",
        "sun_azimuth": 200.0,
        "sun_elevation": 45.0,
    },
]


def generate_crater_patch(seed, size=256, rim_brightness=200, floor_brightness=60,
                          sun_az=135, sun_el=30, roughness=0.6):
    """Generate a realistic synthetic crater patch image."""
    rng = np.random.RandomState(seed)
    h = w = size
    center = size // 2
    radius = int(size * 0.35)

    # Base terrain with fractal noise
    terrain = np.zeros((h, w), dtype=np.float64)
    for octave in range(5):
        freq = 2 ** octave
        amp = roughness ** octave
        noise = rng.randn(max(2, h // freq), max(2, w // freq))
        noise_resized = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        terrain += amp * noise_resized * 15

    # Crater bowl (Gaussian depression)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2).astype(np.float64)
    bowl = np.exp(-0.5 * (dist / (radius * 0.65)) ** 2) * 120
    terrain -= bowl

    # Rim (ring of elevation)
    rim_mask = np.exp(-0.5 * ((dist - radius) / (radius * 0.15)) ** 2)
    terrain += rim_mask * 80

    # Central peak (for larger craters)
    if size > 200:
        peak_r = radius * 0.12
        peak = np.exp(-0.5 * (dist / peak_r) ** 2) * 40
        terrain += peak

    # Secondary craterlets on the floor
    for _ in range(rng.randint(2, 6)):
        cx = center + rng.randint(-radius // 2, radius // 2)
        cy = center + rng.randint(-radius // 2, radius // 2)
        cr = rng.randint(3, max(4, radius // 8))
        d2 = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float64)
        terrain -= np.exp(-0.5 * (d2 / (cr * 0.7)) ** 2) * 20

    # Illumination via Lambertian shading
    az_rad = np.radians(sun_az)
    el_rad = np.radians(sun_el)
    light = np.array([np.cos(el_rad) * np.sin(az_rad),
                      np.cos(el_rad) * np.cos(az_rad),
                      np.sin(el_rad)])

    dy, dx = np.gradient(terrain)
    norm = np.dstack([-dx, -dy, np.ones_like(dx)])
    norms = np.linalg.norm(norm, axis=2, keepdims=True)
    norms[norms == 0] = 1
    norm = norm / norms

    shade = np.clip(np.dot(norm, light), 0.05, 1.0)

    # Albedo variation
    albedo = np.full((h, w), 0.5)
    # Dark floor
    floor_mask = dist < radius * 0.85
    albedo[floor_mask] = floor_brightness / 255.0
    # Bright rim
    rim_band = (dist > radius * 0.85) & (dist < radius * 1.15)
    albedo[rim_band] = rim_brightness / 255.0
    # Ejecta blanket
    ejecta = (dist > radius * 1.15) & (dist < radius * 2.0)
    albedo[ejecta] = 0.55

    img = np.clip(albedo * shade * 255, 0, 255).astype(np.uint8)

    # Add subtle noise
    noise_layer = rng.randint(-8, 8, (h, w)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise_layer, 0, 255).astype(np.uint8)

    # Convert to BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr


def _draw_numbered_craters(image, craters_info):
    """Draw numbered labels on detected craters in an image."""
    vis = image.copy()
    for info in craters_info:
        cx, cy = info['center_x'], info['center_y']
        r = info['radius_px']
        label = info['label']

        # Draw crater circle
        cv2.circle(vis, (cx, cy), r, (0, 255, 200), 2)
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = cx - text_size[0] // 2
        ty = cy - r - 10
        cv2.rectangle(vis, (tx - 4, ty - text_size[1] - 4),
                      (tx + text_size[0] + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 200), 2)
    return vis


def populate_default_catalog():
    """Create the 5 famous craters with synthetic reference images.
    Returns True if catalog was populated, False if already existed."""
    from . import crater_catalog

    if not crater_catalog.catalog_is_empty():
        return False

    for i, crater in enumerate(FAMOUS_CRATERS):
        # Generate two reference patches per crater (different illuminations)
        patch1 = generate_crater_patch(
            seed=i * 100,
            size=300,
            sun_az=crater['sun_azimuth'],
            sun_el=crater['sun_elevation'],
            roughness=0.5 + i * 0.05
        )
        patch2 = generate_crater_patch(
            seed=i * 100 + 50,
            size=300,
            sun_az=(crater['sun_azimuth'] + 90) % 360,
            sun_el=max(15, crater['sun_elevation'] - 10),
            roughness=0.5 + i * 0.05
        )

        # Add crater to catalog
        cid = crater_catalog.add_crater(
            name=crater['name'],
            label=crater['label'],
            lat=crater['latitude'],
            lon=crater['longitude'],
            diameter_km=crater['diameter_km'],
            radius_px=105,
            center_x=150,
            center_y=150,
            source='LROC NAC (Simulated)',
            metadata={'description': crater['description']}
        )

        # Add reference images
        crater_catalog.add_crater_image(
            cid, patch1, image_type='reference',
            sun_az=crater['sun_azimuth'],
            sun_el=crater['sun_elevation'],
            sensor='LROC NAC', confirmed=True
        )
        crater_catalog.add_crater_image(
            cid, patch2, image_type='reference',
            sun_az=(crater['sun_azimuth'] + 90) % 360,
            sun_el=max(15, crater['sun_elevation'] - 10),
            sensor='LROC NAC', confirmed=True
        )

    return True


def detect_and_number_craters(image_gray, min_radius=15, max_radius=120):
    """Detect craters in an image and return numbered detections.
    Returns: list of dicts with center_x, center_y, radius_px, label"""
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_gray)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)

    # Detect circles (craters)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_radius * 2,
        param1=80,
        param2=35,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    detections = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Sort by size (largest first)
        circles = sorted(circles, key=lambda c: c[2], reverse=True)
        for idx, (cx, cy, r) in enumerate(circles):
            detections.append({
                'center_x': int(cx),
                'center_y': int(cy),
                'radius_px': int(r),
                'label': f"C-{idx + 1:03d}",
            })

    return detections


def extract_crater_patch(image, cx, cy, radius, pad=1.5):
    """Extract a square patch around a detected crater for matching.
    Returns cropped image or None if out of bounds."""
    h, w = image.shape[:2]
    half = int(radius * pad)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    patch = image[y1:y2, x1:x2]
    # Resize to standard 300x300 for consistent matching
    patch = cv2.resize(patch, (300, 300), interpolation=cv2.INTER_AREA)
    return patch
