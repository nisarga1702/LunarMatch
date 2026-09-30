import numpy as np
import cv2

def generate_lunar_terrain(width=720, height=520, seed=42, num_craters=25, num_ridges=8):
    """Generate a grayscale lunar terrain image with craters, ridges, and maria.
    Returns: np.ndarray (H, W) float64 heightmap in [0,1] range"""
    np.random.seed(seed)
    # Base terrain with noise
    y, x = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width), dtype=np.float64)
    
    # Add simple fractal noise (sum of octaves)
    for freq in [0.01, 0.05, 0.1]:
        amp = 1.0 / (freq * 100)
        noise = np.random.randn(height, width) * amp
        base += cv2.GaussianBlur(noise, (0, 0), sigmaX=1/freq, sigmaY=1/freq)
        
    base -= base.min()
    base /= (base.max() + 1e-8)
    base *= 0.3 # Background height
    
    # Add craters
    for _ in range(num_craters):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        radius = np.random.uniform(5, 60)
        depth = np.random.uniform(0.1, 0.4)
        
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Crater shape: dip in center, raised rim
        crater = np.zeros_like(base)
        crater[dist < radius] = -depth * (1 - (dist[dist < radius] / radius)**2)
        # Rim
        rim_mask = (dist >= radius) & (dist < radius * 1.5)
        crater[rim_mask] = depth * 0.5 * (1 - (dist[rim_mask] - radius) / (radius * 0.5))
        base += crater
        
    # Add ridges
    for _ in range(num_ridges):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        length = np.random.uniform(50, 200)
        width_r = np.random.uniform(5, 15)
        angle = np.random.uniform(0, 2 * np.pi)
        height_r = np.random.uniform(0.1, 0.3)
        
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Distance to line
        t = (x - cx) * dx + (y - cy) * dy
        proj_x = cx + t * dx
        proj_y = cy + t * dy
        dist_to_line = np.sqrt((x - proj_x)**2 + (y - proj_y)**2)
        
        ridge_mask = (np.abs(t) < length / 2) & (dist_to_line < width_r)
        base[ridge_mask] += height_r * (1 - (dist_to_line[ridge_mask] / width_r)**2)
        
    # Add maria (smooth low areas)
    for _ in range(3):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        radius = np.random.uniform(100, 300)
        depth = np.random.uniform(0.1, 0.2)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        maria = np.zeros_like(base)
        maria[dist < radius] = -depth * (1 - (dist[dist < radius] / radius)**2)
        base += maria
        
    # Normalize
    base = np.clip(base, 0, None)
    base = base / (base.max() + 1e-8)
    return base

def render_illuminated(heightmap, sun_azimuth_deg=120, sun_elevation_deg=35):
    """Render a heightmap with directional illumination (Lambertian shading).
    Returns: np.ndarray (H, W) uint8 grayscale image"""
    h, w = heightmap.shape
    # Compute gradients (normals)
    dzdx = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    dzdy = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    
    # Scale normals
    z_scale = 10.0
    normal = np.dstack((-dzdx * z_scale, -dzdy * z_scale, np.ones((h, w))))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= (norm + 1e-8)
    
    # Sun direction
    az_rad = np.radians(sun_azimuth_deg)
    el_rad = np.radians(sun_elevation_deg)
    sun_dir = np.array([
        np.cos(el_rad) * np.sin(az_rad),
        np.cos(el_rad) * np.cos(az_rad),
        np.sin(el_rad)
    ])
    
    # Lambertian shading
    intensity = np.sum(normal * sun_dir, axis=2)
    intensity = np.clip(intensity, 0, 1)
    
    # Ambient + Diffuse
    ambient = 0.1
    diffuse = 0.9
    rendered = ambient + diffuse * intensity
    
    # Convert to uint8
    rendered_uint8 = (rendered * 255).astype(np.uint8)
    return rendered_uint8

def generate_demo_pair(sun_az_a=120, sun_az_b=240, sun_el_a=35, sun_el_b=45,
                       rotation_deg=7, tx=25, ty=-12, scale=1.0, seed=42):
    """Generate a pair of images: same terrain, different illumination + geometric transform.
    Returns: (img_a, img_b, heightmap, transform_params_dict)
    img_a and img_b are (H,W,3) uint8 BGR images"""
    h, w = 520, 720
    heightmap = generate_lunar_terrain(w, h, seed=seed)
    
    # Render A
    render_a = render_illuminated(heightmap, sun_az_a, sun_el_a)
    
    # Render B
    render_b_base = render_illuminated(heightmap, sun_az_b, sun_el_b)
    
    # Transform B
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    render_b = cv2.warpAffine(render_b_base, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Add noise
    noise_a = np.random.randn(h, w) * 5
    noise_b = np.random.randn(h, w) * 5
    
    render_a = np.clip(render_a.astype(np.float32) + noise_a, 0, 255).astype(np.uint8)
    render_b = np.clip(render_b.astype(np.float32) + noise_b, 0, 255).astype(np.uint8)
    
    # Convert to BGR
    img_a = cv2.cvtColor(render_a, cv2.COLOR_GRAY2BGR)
    img_b = cv2.cvtColor(render_b, cv2.COLOR_GRAY2BGR)
    
    transform_params = {
        'rotation_deg': rotation_deg,
        'tx': tx,
        'ty': ty,
        'scale': scale,
        'matrix': M
    }
    
    return img_a, img_b, heightmap, transform_params
