"""LunarMatch-TGS Core Processing Library"""

from .synthetic_terrain import generate_demo_pair
from .landmark_detector import detect_landmarks, visualize_landmarks, build_landmark_graph, coarse_match_landmarks
from .sun_geometry import (compute_sun_direction, compute_illumination_map, compute_shadow_map,
                           illumination_consistency_score, visualize_sun_direction, compute_confidence_breakdown)
from .feature_matcher import extract_features, match_features_bruteforce, landmark_guided_match, visualize_matches, visualize_keypoints
from .registration import full_pipeline, ransac_filter, apply_registration, compute_registration_metrics
from .crater_catalog import (add_crater, add_crater_image, get_all_craters, get_crater,
                              get_crater_images, get_catalog_stats, confirm_image, reject_image,
                              get_reference_patches, catalog_is_empty)
from .catalog_builder import (populate_default_catalog, detect_and_number_craters,
                               extract_crater_patch, FAMOUS_CRATERS)
