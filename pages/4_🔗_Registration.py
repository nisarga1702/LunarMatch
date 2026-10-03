import streamlit as st
import os
import cv2
import numpy as np

st.set_page_config(page_title="Registration | LunarMatch", page_icon="🔗", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    from core import feature_matcher, registration
except ImportError:
    st.warning("Core modules are still being developed.")
    feature_matcher = None
    registration = None

st.title("🔗 Step 4: Full Pipeline Registration")

if "image_a" not in st.session_state or "image_b" not in st.session_state:
    st.warning("⚠️ No images found. Please go to the Image Input page first.")
    st.stop()

meta = st.session_state.get("metadata", {})

st.sidebar.header("Pipeline Controls")
method = st.sidebar.selectbox("Feature Method", ["SIFT", "AKAZE"])
max_features = st.sidebar.slider("Max Features", 500, 5000, 2000)
ransac_thresh = st.sidebar.slider("RANSAC Threshold", 1.0, 10.0, 3.0)
use_landmarks = st.sidebar.checkbox("Use Landmark Guidance", value=True)
use_sun = st.sidebar.checkbox("Use Sun Geometry Constraint", value=True)
ratio_test = st.sidebar.slider("Ratio Test Threshold", 0.5, 0.9, 0.7)

if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
    if registration:
        progress_text = "Running Registration Pipeline..."
        my_bar = st.progress(0, text=progress_text)
        
        try:
            my_bar.progress(25, text="Extracting features...")
            img_a = st.session_state.image_a
            img_b = st.session_state.image_b
            
            my_bar.progress(50, text="Matching and applying constraints...")
            results = registration.full_pipeline(
                img_a, img_b, 
                meta.get('sun_az_a', 0), meta.get('sun_el_a', 0),
                meta.get('sun_az_b', 0), meta.get('sun_el_b', 0),
                method=method, max_features=max_features, ransac_thresh=ransac_thresh,
                use_landmarks=use_landmarks, use_sun=use_sun, ratio_test=ratio_test
            )
            
            my_bar.progress(85, text="Computing metrics and warping...")
            
            st.session_state.pipeline_results = results
            
            my_bar.progress(100, text="Pipeline Complete!")
            st.success("✅ Registration Pipeline Completed successfully!")
            
        except Exception as e:
            st.error(f"Error running pipeline: {e}")
            my_bar.empty()
    else:
        st.error("Core module 'registration' not found.")

if "pipeline_results" in st.session_state:
    res = st.session_state.pipeline_results
    
    st.markdown("### Metrics Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    metrics = res.get("metrics", {})
    col1.metric("Verified Inliers", metrics.get("inlier_count", 0))
    col2.metric("Inlier Ratio", f"{metrics.get('inlier_ratio', 0.0):.1f}%")
    col3.metric("RMSE (px)", f"{metrics.get('rmse', 0.0):.2f}")
    col4.metric("Confidence", f"{metrics.get('confidence', 0.0):.1f}%")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Feature Detection", "Matching", "RANSAC", "Spatial Distribution", "Registration Result"
    ])
    
    with tab1:
        st.markdown("### Detected Keypoints")
        colA, colB = st.columns(2)
        if feature_matcher and "kp_a_vis" in res and "kp_b_vis" in res:
            with colA:
                st.image(cv2.cvtColor(res["kp_a_vis"], cv2.COLOR_BGR2RGB), caption="Source Features", use_container_width=True)
            with colB:
                st.image(cv2.cvtColor(res["kp_b_vis"], cv2.COLOR_BGR2RGB), caption="Reference Features", use_container_width=True)
        else:
            st.info("Feature visualization not available.")
            
    with tab2:
        st.markdown("### Feature Matches")
        if "matches_vis" in res:
            st.image(cv2.cvtColor(res["matches_vis"], cv2.COLOR_BGR2RGB), use_container_width=True)
            
    with tab3:
        st.markdown("### RANSAC Inliers vs Outliers")
        if "ransac_vis" in res:
            st.image(cv2.cvtColor(res["ransac_vis"], cv2.COLOR_BGR2RGB), use_container_width=True)
            
    with tab4:
        st.markdown("### Spatial Coverage")
        if "coverage_vis" in res:
            st.image(cv2.cvtColor(res["coverage_vis"], cv2.COLOR_BGR2RGB), use_container_width=True)
            st.metric("Spatial Coverage (%)", f"{metrics.get('coverage_pct', 0.0):.1f}%")
            
    with tab5:
        st.markdown("### Registration Result")
        if "warped_image" in res:
            warped = res["warped_image"]
            st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Warped Source Image", use_container_width=True)
            
            st.markdown("### Blend Overlay")
            blend_alpha = st.slider("Blend Ratio", 0.0, 1.0, 0.5)
            if "image_b" in st.session_state:
                img_b_resized = cv2.resize(st.session_state.image_b, (warped.shape[1], warped.shape[0]))
                blended = cv2.addWeighted(warped, blend_alpha, img_b_resized, 1 - blend_alpha, 0)
                st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), caption="Blended Result", use_container_width=True)
