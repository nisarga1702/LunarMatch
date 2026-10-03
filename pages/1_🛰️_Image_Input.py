import streamlit as st
import os
import cv2
import numpy as np
import time

st.set_page_config(page_title="Image Input | LunarMatch", page_icon="🛰️", layout="wide")

# CSS loading boilerplate
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    from core import synthetic_terrain
except ImportError:
    st.warning("Core modules are still being developed. Some features may not work.")
    synthetic_terrain = None

st.title("🛰️ Step 1: Image Input & Metadata")
st.markdown("Load source and reference lunar images with their corresponding metadata.")

mode = st.radio("Input Mode", ["Demo Mode", "Upload Mode"], horizontal=True)

if mode == "Demo Mode":
    st.subheader("Synthetic Terrain Generator")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Source Image (Simulated OHRC)**")
        sun_az_a = st.slider("Sun Azimuth A (°)", 0, 360, 45, key="sun_az_a")
        sun_el_a = st.slider("Sun Elevation A (°)", 0, 90, 30, key="sun_el_a")
    
    with col2:
        st.markdown("**Reference Image (Simulated LROC)**")
        sun_az_b = st.slider("Sun Azimuth B (°)", 0, 360, 135, key="sun_az_b")
        sun_el_b = st.slider("Sun Elevation B (°)", 0, 90, 45, key="sun_el_b")
        
    st.markdown("**Transformation Parameters**")
    col3, col4, col5 = st.columns(3)
    with col3:
        rotation_deg = st.slider("Rotation Angle (°)", -30.0, 30.0, 5.0, key="rot")
    with col4:
        tx = st.slider("Translation X (px)", -50, 50, 15, key="tx")
        ty = st.slider("Translation Y (px)", -50, 50, -10, key="ty")
    with col5:
        scale = st.slider("Scale Factor", 0.5, 2.0, 1.1, key="scale")
        seed = st.number_input("Random Seed", value=42, key="seed")
        
    if st.button("Generate Demo Pair", use_container_width=True):
        if synthetic_terrain:
            with st.spinner("Generating synthetic lunar terrain..."):
                try:
                    img_a, img_b, heightmap, params_dict = synthetic_terrain.generate_demo_pair(
                        sun_az_a, sun_az_b, sun_el_a, sun_el_b, rotation_deg, tx, ty, scale, seed
                    )
                    
                    st.session_state.image_a = img_a
                    st.session_state.image_b = img_b
                    st.session_state.heightmap = heightmap
                    st.session_state.metadata = {
                        "sun_az_a": sun_az_a, "sun_el_a": sun_el_a,
                        "sun_az_b": sun_az_b, "sun_el_b": sun_el_b,
                        "rotation": rotation_deg, "tx": tx, "ty": ty, "scale": scale,
                        "sensor": "OHRC (Simulated)", "reference": "LROC (Simulated)"
                    }
                    st.success("✅ Images Loaded — Proceed to Landmark Detection →")
                except Exception as e:
                    st.error(f"Error generating demo: {e}")
        else:
            st.error("Core module 'synthetic_terrain' not found.")
            
else:
    st.subheader("Upload Custom Images")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Source Image**")
        sensor = st.selectbox("Sensor Type", ["OHRC (0.28m)", "TMC-2 (5m)", "IIRS (80m)"])
        source_file = st.file_uploader("Upload Source Image", type=["png", "jpg", "jpeg", "tif"])
        
    with col2:
        st.markdown("**Reference Image**")
        reference = st.selectbox("Reference Source", ["LRO NAC", "SELENE/Kaguya"])
        ref_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg", "tif"])
        
    with st.expander("📝 Metadata Entry", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Source Metadata**")
            src_sun_az = st.number_input("Source Sun Azimuth", value=45.0)
            src_sun_el = st.number_input("Source Sun Elevation", value=30.0)
        with col4:
            st.markdown("**Reference Metadata**")
            ref_sun_az = st.number_input("Reference Sun Azimuth", value=135.0)
            ref_sun_el = st.number_input("Reference Sun Elevation", value=45.0)
            
    if st.button("Load Images", use_container_width=True):
        if source_file and ref_file:
            try:
                file_bytes_a = np.asarray(bytearray(source_file.read()), dtype=np.uint8)
                img_a = cv2.imdecode(file_bytes_a, cv2.IMREAD_COLOR)
                
                file_bytes_b = np.asarray(bytearray(ref_file.read()), dtype=np.uint8)
                img_b = cv2.imdecode(file_bytes_b, cv2.IMREAD_COLOR)
                
                if img_a is not None and img_b is not None:
                    st.session_state.image_a = img_a
                    st.session_state.image_b = img_b
                    st.session_state.metadata = {
                        "sun_az_a": src_sun_az, "sun_el_a": src_sun_el,
                        "sun_az_b": ref_sun_az, "sun_el_b": ref_sun_el,
                        "rotation": 0, "tx": 0, "ty": 0, "scale": 1,
                        "sensor": sensor, "reference": reference
                    }
                    st.success("✅ Images Loaded — Proceed to Landmark Detection →")
                else:
                    st.error("Failed to decode uploaded images.")
            except Exception as e:
                st.error(f"Error processing images: {e}")
        else:
            st.warning("Please upload both images.")

# Display loaded images
if "image_a" in st.session_state and "image_b" in st.session_state:
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Source: {st.session_state.metadata.get('sensor', 'Source')}")
        st.image(cv2.cvtColor(st.session_state.image_a, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Sun Azimuth: {st.session_state.metadata.get('sun_az_a')}° | Elevation: {st.session_state.metadata.get('sun_el_a')}°")
    with col2:
        st.markdown(f"### Reference: {st.session_state.metadata.get('reference', 'Reference')}")
        st.image(cv2.cvtColor(st.session_state.image_b, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Sun Azimuth: {st.session_state.metadata.get('sun_az_b')}° | Elevation: {st.session_state.metadata.get('sun_el_b')}°")
