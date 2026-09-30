import cv2
import numpy as np

def detect_craters(image_gray, min_radius=8, max_radius=80):
    """Detect circular crater-like features.
    Returns: list of dict with keys 'center' (x,y), 'radius', 'confidence'"""
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_radius*2,
        param1=100, param2=30, minRadius=min_radius, maxRadius=max_radius
    )
    
    craters = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            craters.append({
                'center': (int(x), int(y)),
                'radius': int(r),
                'confidence': 1.0 # Placeholder confidence
            })
    return craters

def detect_edges_ridges(image_gray):
    """Detect strong edges and ridge-like linear features.
    Returns: edge_map (H,W) uint8, list of line segments [(x1,y1,x2,y2,strength)]"""
    blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
    edge_map = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edge_map, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    segments = []
    if lines is not None:
        for line in lines:
            coords = line[0] if line.ndim > 1 else line
            x1, y1, x2, y2 = coords
            strength = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            segments.append((int(x1), int(y1), int(x2), int(y2), float(strength)))
            
    return edge_map, segments

def detect_landmarks(image_gray):
    """Full landmark detection: craters + ridges + strong corners.
    Returns: dict with keys 'craters', 'ridges', 'corners', 'all_points' (list of (x,y))"""
    craters = detect_craters(image_gray)
    _, ridges = detect_edges_ridges(image_gray)
    
    # Corners
    corners_res = cv2.goodFeaturesToTrack(image_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners = []
    if corners_res is not None:
        for pt in corners_res:
            x, y = pt[0]
            corners.append((int(x), int(y)))
            
    all_points = []
    for c in craters:
        all_points.append(c['center'])
    for r in ridges:
        all_points.append(((r[0]+r[2])//2, (r[1]+r[3])//2)) # Midpoint of ridge
    for c in corners:
        all_points.append(c)
        
    return {
        'craters': craters,
        'ridges': ridges,
        'corners': corners,
        'all_points': all_points
    }

def build_landmark_graph(landmarks_a, landmarks_b):
    """Build spatial relationship graphs for two sets of landmarks.
    Compute pairwise distances within each set.
    Returns: (graph_a, graph_b) where each graph is dict with
    'points': list of (x,y), 'distances': NxN ndarray, 'descriptor': per-point feature vector"""
    
    def _build_graph(pts):
        pts = np.array(pts) if len(pts) > 0 else np.zeros((0, 2))
        n = len(pts)
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_mat[i, j] = np.linalg.norm(pts[i] - pts[j])
        
        # Simple descriptor: sorted distances to all other points (normalized)
        descriptors = []
        max_dist = dist_mat.max() if n > 0 and dist_mat.max() > 0 else 1.0
        for i in range(n):
            desc = np.sort(dist_mat[i, :]) / max_dist
            descriptors.append(desc)
            
        return {'points': pts.tolist(), 'distances': dist_mat, 'descriptor': descriptors}

    return _build_graph(landmarks_a), _build_graph(landmarks_b)

def coarse_match_landmarks(graph_a, graph_b, tolerance=0.15):
    """Match landmarks between two graphs using distance-ratio consistency.
    Returns: list of (idx_a, idx_b, confidence) tuples"""
    matches = []
    desc_a = graph_a['descriptor']
    desc_b = graph_b['descriptor']
    
    if not desc_a or not desc_b:
        return matches
        
    for i, da in enumerate(desc_a):
        best_j = -1
        best_score = float('inf')
        
        for j, db in enumerate(desc_b):
            # Compare descriptors (sorted normalized distances)
            min_len = min(len(da), len(db))
            if min_len < 3:
                continue
            
            # Compute distance between descriptors
            diff = np.abs(da[:min_len] - db[:min_len])
            score = np.mean(diff)
            
            if score < best_score:
                best_score = score
                best_j = j
                
        if best_j != -1 and best_score < tolerance:
            confidence = 1.0 - (best_score / tolerance)
            matches.append((i, best_j, confidence))
            
    return matches

def visualize_landmarks(image, landmarks, color_craters=(0,255,200), color_ridges=(255,100,50)):
    """Draw detected landmarks on image. Returns annotated BGR image."""
    res = image.copy()
    for c in landmarks.get('craters', []):
        cv2.circle(res, c['center'], c['radius'], color_craters, 2)
        cv2.circle(res, c['center'], 2, (0, 0, 255), -1)
        
    for r in landmarks.get('ridges', []):
        x1, y1, x2, y2, _ = r
        cv2.line(res, (x1, y1), (x2, y2), color_ridges, 2)
        
    for c in landmarks.get('corners', []):
        cv2.circle(res, c, 3, (255, 0, 255), -1)
        
    return res

def visualize_landmark_graph(image, landmarks, connections=True):
    """Draw landmarks with connecting graph edges. Returns annotated BGR image."""
    res = image.copy()
    pts = landmarks.get('all_points', [])
    if connections and len(pts) > 1:
        # Draw some edges (e.g., minimum spanning tree or k-NN, here just draw closest neighbor)
        for i, pt1 in enumerate(pts):
            best_dist = float('inf')
            best_j = -1
            for j, pt2 in enumerate(pts):
                if i == j: continue
                d = (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j != -1:
                cv2.line(res, tuple(map(int, pt1)), tuple(map(int, pts[best_j])), (100, 100, 100), 1)
                
    for p in pts:
        cv2.circle(res, tuple(map(int, p)), 3, (0, 255, 255), -1)
    return res
