import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd

from core.crater_catalog import (
    get_reference_patches, get_all_craters, train_and_add_to_album, get_crater_images
)
from core.catalog_builder import (
    detect_and_number_craters, extract_crater_patch, generate_crater_patch, populate_default_catalog
)
from core.feature_matcher import extract_features, match_features_bruteforce, visualize_matches
from core.registration import ransac_filter, compute_registration_metrics

st.set_page_config(page_title="Match & Train | LunarMatch", page_icon="🛰️", layout="wide")

# Ensure default catalog is populated
populate_default_catalog()

st.title("🛰️ Source Image Search & Reference Album Training")
st.caption("Upload source lunar imagery to search across all reference albums and train the descriptor model.")

st.markdown("""
<div class='glass-card'>
    <h4>⚡ Interactive Album Training Workflow:</h4>
    <ol>
        <li><b>Upload Source Image:</b> Upload a Chandrayaan-2 (OHRC/TMC-2/IIRS) observation image.</li>
        <li><b>Global Database Search:</b> The system scans all reference images in the catalog database.</li>
        <li><b>Album Candidate Report:</b> View match scores, feature alignments, and candidate albums.</li>
        <li><b>Train Model ("Click YES"):</b> Click <b>"YES - Add to Album & Retrain"</b> to add the image to the reference album and train the model!</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.write("")

# ── Step 1: Input Source Image Selection ──
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    st.subheader("1. Select Source Image")
    source_type = st.radio("Choose Source Type:", ["Synthetic Demo Scene (Fast test)", "Upload New Image File"])
    
    img_bgr = None
    if source_type == "Synthetic Demo Scene (Fast test)":
        target_crater_name = st.selectbox("Select target crater for demo observation:", ["Tycho", "Copernicus", "Aristarchus", "Kepler", "Plato"])
        crater_seeds = {"Tycho": 0, "Copernicus": 100, "Aristarchus": 200, "Kepler": 300, "Plato": 400}
        seed = crater_seeds.get(target_crater_name, 0)
        # Generate simulated CH2 capture with illumination variation
        patch = generate_crater_patch(seed=seed + 15, size=350, sun_az=150, sun_el=28, roughness=0.55)
        img_bgr = patch
        st.info(f"Simulated Chandrayaan-2 observation image generated for **{target_crater_name}**.")
    else:
        uploaded = st.file_uploader("Upload Chandrayaan-2 Source Image (PNG/JPG/TIF):", type=["png", "jpg", "jpeg", "tif"])
        if uploaded is not None:
            buf = np.frombuffer(uploaded.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

with col_input2:
    st.subheader("Source Image Preview")
    if img_bgr is not None:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Source Image", use_container_width=True)
    else:
        st.warning("Please upload an image or choose a demo scene.")

st.write("---")

# ── Step 2: Run Search Across ALL Reference Albums ──
if img_bgr is not None:
    st.subheader("2. Search Across All Reference Albums")
    
    feat_method = st.selectbox("Feature Extraction Algorithm:", ["SIFT", "AKAZE"], index=0)
    
    if st.button("🔍 Search Database & Generate Album Report", type="primary"):
        with st.spinner("Scanning all reference albums and extracting keypoint descriptors..."):
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Extract features from input image
            kp_input, desc_input = extract_features(img_gray, method=feat_method.lower())
            
            # Fetch all reference patches across database
            ref_patches = get_reference_patches()
            
            if not ref_patches:
                st.error("Reference database is empty. Please populate catalog on Page 1.")
            elif desc_input is None or len(kp_input) < 4:
                st.error("Could not extract sufficient keypoints from source image.")
            else:
                match_results = []
                
                # Compare source image against each reference patch
                for ref in ref_patches:
                    ref_img_gray = cv2.cvtColor(ref['image'], cv2.COLOR_BGR2GRAY)
                    kp_ref, desc_ref = extract_features(ref_img_gray, method=feat_method.lower())
                    
                    if desc_ref is None or len(kp_ref) < 4:
                        continue
                        
                    raw_matches = match_features_bruteforce(desc_input, desc_ref)
                    src_pts, dst_pts, inlier_matches, H, mask = ransac_filter(kp_input, kp_ref, raw_matches)
                    
                    num_inliers = len(inlier_matches)
                    inlier_ratio = (num_inliers / len(raw_matches)) if raw_matches else 0.0
                    confidence = min(0.99, (num_inliers / 30.0) * 0.6 + inlier_ratio * 0.4)
                    
                    vis_img = visualize_matches(img_bgr, ref['image'], kp_input, kp_ref, raw_matches, mask)
                    
                    match_results.append({
                        'crater_id': ref['crater_id'],
                        'crater_name': ref['name'],
                        'label': ref['label'],
                        'ref_image': ref['image'],
                        'image_type': ref.get('image_type', 'reference'),
                        'raw_matches': len(raw_matches),
                        'inliers': num_inliers,
                        'confidence': round(confidence * 100, 1),
                        'vis_image': vis_img,
                        'homography': H
                    })
                    
                # Group and sort by highest confidence per crater
                match_results.sort(key=lambda x: x['confidence'], reverse=True)
                
                st.session_state['new_image_bgr'] = img_bgr
                st.session_state['album_match_results'] = match_results
                st.session_state['top_match'] = match_results[0] if match_results else None
                
                st.success(f"Search Complete! Scanned against {len(ref_patches)} reference items across all albums.")

# ── Step 3: Reference Album Report & Interactive Training ("Click YES") ──
if 'album_match_results' in st.session_state and st.session_state['album_match_results']:
    match_results = st.session_state['album_match_results']
    source_img = st.session_state.get('new_image_bgr')
    
    st.markdown("---")
    st.subheader("🖼️ Reference Album Candidate Match Report")
    st.write("Review candidate album matches below. Click **'YES - Add to Album & Train Model'** on the matching crater album to retrain the descriptor model.")
    
    # Render each matching album as an interactive card
    for idx, candidate in enumerate(match_results[:3]): # Top 3 album candidates
        st.markdown(f"### Candidate #{idx+1}: Album **{candidate['crater_name']}** (`{candidate['label']}`)")
        
        card_col1, card_col2 = st.columns([1, 2])
        
        with card_col1:
            st.markdown(f"""
            <div class='glass-card' style='border-left: 5px solid {"#00ff9d" if candidate["confidence"] > 60 else "#00b4d8"};'>
                <h4>🖼️ Album: {candidate['crater_name']}</h4>
                <p><b>Match Confidence:</b> <span style='font-size:1.4rem; color: #00ff9d;'>{candidate['confidence']}%</span></p>
                <p><b>RANSAC Inliers:</b> {candidate['inliers']} / {candidate['raw_matches']}</p>
                <p><b>Crater ID:</b> #{candidate['crater_id']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Training YES Button
            yes_key = f"yes_train_btn_{candidate['crater_id']}_{idx}"
            if st.button(f"✅ YES — Add to Album #{candidate['crater_id']} ({candidate['crater_name']}) & Retrain Model", key=yes_key, type="primary"):
                if source_img is not None:
                    train_and_add_to_album(
                        crater_id=candidate['crater_id'],
                        image=source_img,
                        image_type='trained_reference',
                        sensor='Chandrayaan-2 OHRC',
                        confirmed=True
                    )
                    st.session_state['selected_album_id'] = candidate['crater_id']
                    st.session_state['last_trained_album'] = candidate['crater_name']
                    st.success(f"🎉 YES confirmed! Image added to **Album #{candidate['crater_id']} ({candidate['crater_name']})**. Descriptor database updated & trained!")
                    st.balloons()
                    
        with card_col2:
            st.write("**Feature Alignment (Source vs Album Reference):**")
            if candidate['vis_image'] is not None:
                st.image(cv2.cvtColor(candidate['vis_image'], cv2.COLOR_BGR2RGB), caption=f"Keypoint Correspondences with Album {candidate['crater_name']}", use_container_width=True)
                
        st.write("---")

    # Table of all matches
    st.subheader("📋 All Album Match Scores")
    df_rank = pd.DataFrame([
        {
            "Album ID": f"#{r['crater_id']}",
            "Crater Label": r['label'],
            "Crater Name": r['crater_name'],
            "Confidence": f"{r['confidence']}%",
            "Inlier Matches": r['inliers'],
            "Raw Matches": r['raw_matches']
        }
        for r in match_results
    ])
    st.table(df_rank)
