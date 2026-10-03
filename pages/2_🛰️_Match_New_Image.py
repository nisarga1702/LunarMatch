import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd

from core.crater_catalog import get_reference_patches, get_all_craters
from core.catalog_builder import detect_and_number_craters, extract_crater_patch, generate_crater_patch, populate_default_catalog
from core.feature_matcher import extract_features, match_features_bruteforce, visualize_matches
from core.registration import ransac_filter, compute_registration_metrics

st.set_page_config(page_title="Match New Image | LunarMatch", page_icon="🛰️", layout="wide")

# Ensure default catalog is populated
populate_default_catalog()

st.title("🛰️ Match New Chandrayaan-2 Image to Catalog")
st.caption("Upload a new observation image to identify which catalog crater it belongs to.")

st.markdown("""
<div class='glass-card'>
    <h4>How it works:</h4>
    <ol>
        <li>Upload a new Chandrayaan-2 (OHRC/TMC) lunar image or generate a test scene.</li>
        <li>The system detects craters and extracts candidate features.</li>
        <li>Candidates are matched against the <b>Crater Catalog Reference Database</b>.</li>
        <li>The system identifies the exact matching catalog crater (e.g. Tycho, Copernicus) with confidence scores.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.write("")

# ── Step 1: Input Image Selection ──
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    st.subheader("1. Select Input Source")
    source_type = st.radio("Choose Image Source:", ["Synthetic Demo Scene (Fast test)", "Upload New Image File"])
    
    img_bgr = None
    if source_type == "Synthetic Demo Scene (Fast test)":
        target_crater_name = st.selectbox("Select target famous crater for demo image:", ["Tycho", "Copernicus", "Aristarchus", "Kepler", "Plato"])
        crater_seeds = {"Tycho": 0, "Copernicus": 100, "Aristarchus": 200, "Kepler": 300, "Plato": 400}
        seed = crater_seeds.get(target_crater_name, 0)
        # Generate slightly rotated/illuminated version simulating CH2 capture
        patch = generate_crater_patch(seed=seed + 15, size=350, sun_az=150, sun_el=28, roughness=0.55)
        img_bgr = patch
        st.info(f"Generated simulated Chandrayaan-2 observation image for **{target_crater_name}**.")
    else:
        uploaded = st.file_uploader("Upload Chandrayaan-2 Image (PNG/JPG):", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            buf = np.frombuffer(uploaded.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

with col_input2:
    st.subheader("Input Image Preview")
    if img_bgr is not None:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded / Generated New Observation", use_container_width=True)
    else:
        st.warning("Please upload an image or choose a demo scene to proceed.")

st.write("---")

# ── Step 2: Run Catalog Matching ──
if img_bgr is not None:
    st.subheader("2. Run Crater Matching Pipeline")
    
    feat_method = st.selectbox("Feature Extractor:", ["SIFT", "AKAZE"], index=0)
    
    if st.button("🚀 Run Crater Catalog Matching", type="primary"):
        with st.spinner("Detecting craters and matching against Crater Catalog Database..."):
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Extract features from input image
            kp_input, desc_input = extract_features(img_gray, method=feat_method.lower())
            
            # Fetch catalog reference patches
            ref_patches = get_reference_patches()
            
            if not ref_patches:
                st.error("Catalog database is empty. Please populate catalog on Page 1.")
            elif desc_input is None or len(kp_input) < 4:
                st.error("Could not extract sufficient keypoints from input image.")
            else:
                match_results = []
                
                # Match against each reference crater patch
                for ref in ref_patches:
                    ref_img_gray = cv2.cvtColor(ref['image'], cv2.COLOR_BGR2GRAY)
                    kp_ref, desc_ref = extract_features(ref_img_gray, method=feat_method.lower())
                    
                    if desc_ref is None or len(kp_ref) < 4:
                        continue
                        
                    raw_matches = match_features_bruteforce(desc_input, desc_ref)
                    
                    # Apply RANSAC filtering
                    src_pts, dst_pts, inlier_matches, H, mask = ransac_filter(kp_input, kp_ref, raw_matches)
                    
                    num_inliers = len(inlier_matches)
                    inlier_ratio = (num_inliers / len(raw_matches)) if raw_matches else 0.0
                    confidence = min(0.99, (num_inliers / 30.0) * 0.6 + inlier_ratio * 0.4)
                    
                    # Visualization
                    vis_img = visualize_matches(img_bgr, ref['image'], kp_input, kp_ref, raw_matches, mask)
                    
                    match_results.append({
                        'crater_id': ref['crater_id'],
                        'crater_name': ref['name'],
                        'label': ref['label'],
                        'ref_image': ref['image'],
                        'raw_matches': len(raw_matches),
                        'inliers': num_inliers,
                        'confidence': round(confidence * 100, 1),
                        'vis_image': vis_img,
                        'homography': H
                    })
                    
                # Sort by confidence
                match_results.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Store in session state
                st.session_state['new_image_bgr'] = img_bgr
                st.session_state['match_results'] = match_results
                st.session_state['top_match'] = match_results[0] if match_results else None
                
                st.success(f"Matching Complete! Evaluated against {len(ref_patches)} reference catalog entries.")

# ── Display Results ──
if 'match_results' in st.session_state and st.session_state['match_results']:
    match_results = st.session_state['match_results']
    top = match_results[0]
    
    st.markdown("---")
    st.subheader("🎯 Identification Result")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown(f"""
        <div class='glass-card' style='border-left: 5px solid var(--accent);'>
            <h3>Match Identified:</h3>
            <h2>{top['crater_name']} ({top['label']})</h2>
            <p><b>Confidence Score:</b> <span style='font-size:1.5rem; color:#00ff9d;'>{top['confidence']}%</span></p>
            <p><b>Inliers / Raw Matches:</b> {top['inliers']} / {top['raw_matches']}</p>
            <p><b>Catalog Crater ID:</b> #{top['crater_id']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("👉 Go to **Page 3 (Sun Geometry)** to analyze illumination, or **Page 4 (Confirm & Update)** to confirm and save this image to the catalog database!")
        
    with res_col2:
        st.write("**Feature Alignment Visualization (Input vs Catalog Reference):**")
        if top['vis_image'] is not None:
            st.image(cv2.cvtColor(top['vis_image'], cv2.COLOR_BGR2RGB), caption=f"Keypoint Correspondences with {top['crater_name']}", use_container_width=True)

    # Leaderboard of top matches
    st.subheader("Crater Catalog Candidate Rankings")
    rank_data = []
    for r in match_results:
        rank_data.append({
            "Crater Label": r['label'],
            "Crater Name": r['crater_name'],
            "Confidence": f"{r['confidence']}%",
            "Inlier Matches": r['inliers'],
            "Raw Matches": r['raw_matches']
        })
    st.table(pd.DataFrame(rank_data))
