import streamlit as st
import cv2
import numpy as np
import pandas as pd
from core.sun_geometry import (
    compute_sun_direction, compute_illumination_map, compute_shadow_map,
    illumination_consistency_score, visualize_sun_direction, compute_confidence_breakdown
)
from core.catalog_builder import populate_default_catalog

st.set_page_config(page_title="Sun Geometry | LunarMatch", page_icon="☀️", layout="wide")
populate_default_catalog()

st.title("☀️ Sun Geometry & Illumination Analysis")
st.caption("Temporal and Solar Illumination Consistency Verification")

if 'top_match' not in st.session_state or st.session_state['top_match'] is None:
    st.warning("⚠️ Please run matching on Page 2 first to select a candidate image for solar geometry analysis.")
    st.stop()

top_match = st.session_state['top_match']

st.markdown(f"""
<div class='glass-card'>
    <h3>Target Analysis Candidate: <b>{top_match['crater_name']} ({top_match['label']})</b></h3>
    <p>Feature Match Confidence: <b>{top_match['confidence']}%</b></p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Inputs for Sun Geometry
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("Input Image Sun Position (Chandrayaan-2)")
    sun_az_a = st.slider("Input Sun Azimuth (°):", 0.0, 360.0, 150.0, 5.0)
    sun_el_a = st.slider("Input Sun Elevation (°):", 5.0, 85.0, 28.0, 1.0)

with col_g2:
    st.subheader("Reference Image Sun Position (LROC Catalog)")
    sun_az_b = st.slider("Reference Sun Azimuth (°):", 0.0, 360.0, 135.0, 5.0)
    sun_el_b = st.slider("Reference Sun Elevation (°):", 5.0, 85.0, 30.0, 1.0)

st.write("---")

# Compute illumination maps
if 'new_image_bgr' in st.session_state:
    img_a = st.session_state['new_image_bgr']
    img_b = top_match['ref_image']
    
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    
    illum_a = compute_illumination_map(gray_a, sun_az_a, sun_el_a)
    illum_b = compute_illumination_map(gray_b, sun_az_b, sun_el_b)
    
    shad_a = compute_shadow_map(gray_a)
    shad_b = compute_shadow_map(gray_b)
    
    sun_score = illumination_consistency_score(illum_a, illum_b, shad_a, shad_b)
    
    # Confidence breakdown
    geom_score = min(1.0, top_match['inliers'] / 30.0)
    struct_score = top_match['confidence'] / 100.0
    
    overall_conf, breakdown = compute_confidence_breakdown(struct_score, geom_score, sun_score)
    
    st.subheader("Illumination & Shadow Maps")
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        st.image(illum_a, caption="Input Illumination Shading Map", use_container_width=True)
        st.image(shad_a, caption="Input Shadow Mask", use_container_width=True)
        
    with vis_col2:
        st.image(illum_b, caption="Reference Illumination Shading Map", use_container_width=True)
        st.image(shad_b, caption="Reference Shadow Mask", use_container_width=True)
        
    st.write("---")
    st.subheader("Combined Confidence Assessment")
    
    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
    with c_col1:
        st.metric("Structure Score", f"{breakdown['structure_score'] * 100:.1f}%")
    with c_col2:
        st.metric("Geometry Score", f"{breakdown['geometry_score'] * 100:.1f}%")
    with c_col3:
        st.metric("Sun Illumination Score", f"{breakdown['sun_score'] * 100:.1f}%")
    with c_col4:
        st.metric("Overall Match Confidence", f"{overall_conf * 100:.1f}%", delta="HIGH CONFIDENCE" if overall_conf > 0.6 else "MEDIUM CONFIDENCE")
        
    st.info("👉 Ready to link this image to the catalog? Proceed to **Page 4 (Confirm & Update)**.")
