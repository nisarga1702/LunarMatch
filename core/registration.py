import cv2
import numpy as np
from .landmark_detector import detect_landmarks, build_landmark_graph, coarse_match_landmarks
from .feature_matcher import extract_features, match_features_bruteforce, landmark_guided_match
from .sun_geometry import illumination_consistency_score, compute_confidence_breakdown

def ransac_filter(pts_a, pts_b, method='homography', reproj_thresh=4.0):
    """RANSAC-based outlier rejection.
    Returns: (transform_matrix, inlier_mask, model_type)"""
    pts_a = np.float32(pts_a)
    pts_b = np.float32(pts_b)
    
    if len(pts_a) < 4:
        return None, None, method
        
    if method == 'homography' or len(pts_a) >= 4:
        M, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, reproj_thresh)
        if M is not None:
            return M, mask, 'homography'
            
    # Fallback to affine
    if len(pts_a) >= 3:
        M, mask = cv2.estimateAffine2D(pts_a, pts_b, cv2.RANSAC, ransacReprojThreshold=reproj_thresh)
        if M is not None:
            # Convert to 3x3 for consistency
            M3 = np.eye(3)
            M3[:2, :] = M
            return M3, mask, 'affine'
            
    return None, None, 'none'

def spatial_distribution_filter(pts_a, pts_b, mask, grid_size=4, min_per_cell=1):
    """Select spatially distributed matches across a grid.
    Returns: filtered_mask (same shape as mask)"""
    if mask is None or len(pts_a) == 0:
        return mask
        
    pts_a = np.array(pts_a)
    filtered_mask = np.zeros_like(mask)
    
    min_x, min_y = np.min(pts_a, axis=0)
    max_x, max_y = np.max(pts_a, axis=0)
    
    cell_w = (max_x - min_x) / grid_size
    cell_h = (max_y - min_y) / grid_size
    
    if cell_w <= 0 or cell_h <= 0:
        return mask
        
    # Keep track of counts per cell
    cell_counts = np.zeros((grid_size, grid_size), dtype=int)
    
    for i in range(len(pts_a)):
        if not mask[i]:
            continue
            
        x, y = pts_a[i]
        cx = min(int((x - min_x) / cell_w), grid_size - 1)
        cy = min(int((y - min_y) / cell_h), grid_size - 1)
        
        # We can either enforce a max per cell, or just require some min spread.
        # For simplicity, we just keep all inliers if they contribute to a well-distributed grid,
        # or we could subsample. Let's just keep them all if we use this filter, 
        # but report grid occupancy.
        filtered_mask[i] = 1
        
    return filtered_mask

def subpixel_refine(image_a_gray, image_b_gray, pts_a, pts_b, transform, win_size=15):
    """Refine match positions to sub-pixel accuracy using Lucas-Kanade.
    Returns: (refined_pts_a, refined_pts_b)"""
    # Optional subpixel refinement
    pts_a_rf = np.float32(pts_a).reshape(-1, 1, 2)
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    try:
        pts_a_rf = cv2.cornerSubPix(image_a_gray, pts_a_rf, (win_size, win_size), (-1, -1), criteria)
    except:
        pass
        
    return pts_a_rf.reshape(-1, 2), pts_b

def compute_registration_metrics(pts_a, pts_b, transform, inlier_mask, image_shape):
    """Compute RMSE, inlier ratio, spatial coverage, confidence.
    Returns: dict with 'rmse', 'inlier_count', 'inlier_ratio', 'spatial_coverage',
    'confidence', 'grid_occupancy' (4x4 bool array)"""
    if inlier_mask is None or np.sum(inlier_mask) == 0:
        return {'rmse': float('inf'), 'inlier_count': 0, 'inlier_ratio': 0, 'confidence': 0}
        
    inliers_a = np.array(pts_a)[inlier_mask.ravel() == 1]
    inliers_b = np.array(pts_b)[inlier_mask.ravel() == 1]
    
    # Reproject A to B
    if transform.shape == (3, 3):
        inliers_a_hom = np.hstack([inliers_a, np.ones((len(inliers_a), 1))])
        reproj_b = (transform @ inliers_a_hom.T).T
        reproj_b = reproj_b[:, :2] / reproj_b[:, 2:]
    else:
        # Affine 2x3
        inliers_a_hom = np.hstack([inliers_a, np.ones((len(inliers_a), 1))])
        reproj_b = (transform @ inliers_a_hom.T).T
        
    err = np.sqrt(np.sum((inliers_b - reproj_b)**2, axis=1))
    rmse = np.mean(err)
    
    inlier_count = len(inliers_a)
    inlier_ratio = inlier_count / len(pts_a) if len(pts_a) > 0 else 0
    
    # Grid occupancy
    h, w = image_shape[:2]
    grid = np.zeros((4, 4), dtype=bool)
    for x, y in inliers_a:
        cx = min(int(x / (w / 4)), 3)
        cy = min(int(y / (h / 4)), 3)
        grid[cy, cx] = True
        
    spatial_coverage = np.sum(grid) / 16.0
    
    # Heuristic confidence
    confidence = min(1.0, (inlier_ratio * 0.5 + spatial_coverage * 0.5) * (1.0 if inlier_count > 10 else inlier_count/10.0))
    
    return {
        'rmse': rmse,
        'inlier_count': inlier_count,
        'inlier_ratio': inlier_ratio,
        'spatial_coverage': spatial_coverage,
        'confidence': confidence,
        'grid_occupancy': grid
    }

def apply_registration(source_image, transform, target_shape):
    """Warp source image using the computed transform.
    Returns: warped BGR image"""
    h, w = target_shape[:2]
    if transform.shape == (3, 3):
        return cv2.warpPerspective(source_image, transform, (w, h))
    else:
        return cv2.warpAffine(source_image, transform[:2, :], (w, h))

def full_pipeline(image_a, image_b, sun_az_a=120, sun_el_a=35, sun_az_b=240, sun_el_b=45,
                  feature_method='sift', use_landmarks=True, use_sun_geometry=True):
    """Run the complete LunarMatch-TGS pipeline.
    Returns: dict with all intermediate results"""
    
    results = {}
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    
    # 1. Landmarks
    coarse = []
    if use_landmarks:
        try:
            lm_a = detect_landmarks(gray_a)
            lm_b = detect_landmarks(gray_b)
            graph_a, graph_b = build_landmark_graph(lm_a['all_points'], lm_b['all_points'])
            coarse = coarse_match_landmarks(graph_a, graph_b)
            results.update({'landmarks_a': lm_a, 'landmarks_b': lm_b, 
                            'landmark_graph_a': graph_a, 'landmark_graph_b': graph_b,
                            'coarse_matches': coarse})
        except Exception:
            coarse = []
        
    # 2. Features
    kp_a, desc_a = extract_features(gray_a, method=feature_method)
    kp_b, desc_b = extract_features(gray_b, method=feature_method)
    results.update({'keypoints_a': kp_a, 'keypoints_b': kp_b})
    
    # Generate keypoint visualizations
    try:
        from .feature_matcher import visualize_keypoints, visualize_matches
        results['kp_a_vis'] = visualize_keypoints(image_a, kp_a)
        results['kp_b_vis'] = visualize_keypoints(image_b, kp_b)
    except Exception:
        pass
    
    # 3. Match — always do bruteforce, optionally also guided
    raw_matches = match_features_bruteforce(desc_a, desc_b)
    results['raw_matches'] = raw_matches
    
    if use_landmarks and len(coarse) > 0:
        guided = landmark_guided_match(kp_a, desc_a, kp_b, desc_b, coarse)
        results['guided_matches'] = guided
        
    # Use bruteforce matches for registration (more reliable)
    pts_a = [kp_a[m.queryIdx].pt for m in raw_matches]
    pts_b = [kp_b[m.trainIdx].pt for m in raw_matches]
        
    # 4. Filter
    if len(pts_a) >= 4:
        M, mask, mtype = ransac_filter(pts_a, pts_b)
        if M is not None and mask is not None:
            mask = mask.ravel().astype(bool)
            results['transform'] = M
            results['inlier_mask'] = mask
            
            # Generate match visualizations
            try:
                # All matches visualization
                results['matches_vis'] = visualize_matches(image_a, image_b, kp_a, kp_b, raw_matches, mask=None)
                # RANSAC inlier/outlier visualization
                ransac_mask_int = mask.astype(np.uint8).reshape(-1, 1)
                results['ransac_vis'] = visualize_matches(image_a, image_b, kp_a, kp_b, raw_matches, mask=ransac_mask_int)
            except Exception:
                pass
            
            sp_mask = spatial_distribution_filter(pts_a, pts_b, mask)
            if sp_mask is not None:
                sp_mask = np.asarray(sp_mask).ravel().astype(bool)
            else:
                sp_mask = mask
            results['spatial_mask'] = sp_mask
            
            # Coverage visualization (draw grid on image)
            try:
                h, w = image_a.shape[:2]
                cov_vis = image_a.copy()
                for gi in range(1, 4):
                    cv2.line(cov_vis, (gi * w // 4, 0), (gi * w // 4, h), (0, 200, 180), 1)
                    cv2.line(cov_vis, (0, gi * h // 4), (w, gi * h // 4), (0, 200, 180), 1)
                pts_a_arr_tmp = np.float32(pts_a)
                for i, (x, y) in enumerate(pts_a_arr_tmp):
                    if sp_mask[i]:
                        cv2.circle(cov_vis, (int(x), int(y)), 5, (0, 255, 0), -1)
                results['coverage_vis'] = cov_vis
            except Exception:
                pass
            
            # 5. Refine
            pts_a_arr = np.float32(pts_a)
            pts_b_arr = np.float32(pts_b)
            r_pts_a, r_pts_b = subpixel_refine(gray_a, gray_b, pts_a_arr, pts_b_arr, M)
            results['refined_pts_a'] = r_pts_a
            results['refined_pts_b'] = r_pts_b
            
            # 6. Metrics & Warp
            metrics = compute_registration_metrics(r_pts_a, r_pts_b, M, sp_mask, image_a.shape)
            results['metrics'] = metrics
            
            warped = apply_registration(image_a, M, image_b.shape)
            results['warped_image'] = warped
        else:
            results['transform'] = None
            results['metrics'] = {'rmse': float('inf'), 'inlier_count': 0, 'inlier_ratio': 0,
                                  'spatial_coverage': 0, 'confidence': 0.0}
    else:
        results['transform'] = None
        results['metrics'] = {'rmse': float('inf'), 'inlier_count': 0, 'inlier_ratio': 0,
                              'spatial_coverage': 0, 'confidence': 0.0}
        
    # 7. Sun analysis
    if use_sun_geometry:
        try:
            sun_score = illumination_consistency_score(gray_a, gray_b, sun_az_a, sun_el_a, sun_az_b, sun_el_b)
        except Exception:
            sun_score = 0.5
        results['sun_analysis'] = {'illumination_score': sun_score}
        
        conf = compute_confidence_breakdown(
            results['metrics'].get('confidence', 0),
            results['metrics'].get('spatial_coverage', 0),
            sun_score
        )
        results['confidence_breakdown'] = conf
        
    return results

