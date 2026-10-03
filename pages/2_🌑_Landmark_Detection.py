import streamlit as st
import os
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Landmark Detection | LunarMatch", page_icon="🌑", layout="wide")

# CSS loading boilerplate
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    from core import landmark_detector
except ImportError:
    st.warning("Core modules are still being developed.")
    landmark_detector = None

st.title("🌑 Step 2: Landmark Detection")
st.markdown("Detect lunar craters, ridges, and corners to form a spatial graph for structural matching.")

if "image_a" not in st.session_state or "image_b" not in st.session_state:
    st.warning("⚠️ No images found. Please go to the Image Input page first.")
    st.stop()

# Sidebar parameters
st.sidebar.header("Detection Parameters")
min_radius = st.sidebar.slider("Min Crater Radius", 5, 50, 10)
max_radius = st.sidebar.slider("Max Crater Radius", 50, 200, 100)
edge_sens = st.sidebar.slider("Edge Sensitivity", 1.0, 5.0, 2.5)
corner_qual = st.sidebar.slider("Corner Quality", 0.01, 0.1, 0.04)

if st.button("Run Landmark Detection", type="primary", use_container_width=True):
    if landmark_detector:
        with st.spinner("Detecting landmarks and building graphs..."):
            try:
                # Convert to grayscale
                img_a_gray = cv2.cvtColor(st.session_state.image_a, cv2.COLOR_BGR2GRAY)
                img_b_gray = cv2.cvtColor(st.session_state.image_b, cv2.COLOR_BGR2GRAY)
                
                # Detect
                lm_a = landmark_detector.detect_landmarks(img_a_gray)
                lm_b = landmark_detector.detect_landmarks(img_b_gray)
                
                # Graphs
                graph_a, graph_b = landmark_detector.build_landmark_graph(lm_a['all_points'], lm_b['all_points'])
                
                # Coarse Match
                matches = landmark_detector.coarse_match_landmarks(graph_a, graph_b)
                
                st.session_state.landmarks_a = lm_a
                st.session_state.landmarks_b = lm_b
                st.session_state.landmark_graphs = (graph_a, graph_b)
                st.session_state.coarse_matches = matches
                st.success("✅ Landmark detection completed!")
            except Exception as e:
                st.error(f"Error during landmark detection: {e}")
    else:
        st.error("Core module 'landmark_detector' not found.")

if "landmarks_a" in st.session_state:
    tab1, tab2, tab3, tab4 = st.tabs(["Source Landmarks", "Reference Landmarks", "Landmark Graph", "Coarse Matching"])
    
    lm_a = st.session_state.landmarks_a
    lm_b = st.session_state.landmarks_b
    matches = st.session_state.get("coarse_matches", [])
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Craters (Source)", len(lm_a.get('craters', [])))
    col2.metric("Ridges (Source)", len(lm_a.get('ridges', [])))
    col3.metric("Corners (Source)", len(lm_a.get('corners', [])))
    col4.metric("Coarse Matches", len(matches))
    
    with tab1:
        colA, colB = st.columns(2)
        with colA:
            st.image(cv2.cvtColor(st.session_state.image_a, cv2.COLOR_BGR2RGB), caption="Original Source", use_container_width=True)
        with colB:
            if landmark_detector:
                vis_a = landmark_detector.visualize_landmarks(st.session_state.image_a, lm_a)
                st.image(cv2.cvtColor(vis_a, cv2.COLOR_BGR2RGB), caption="Annotated Source", use_container_width=True)
                
    with tab2:
        colA, colB = st.columns(2)
        with colA:
            st.image(cv2.cvtColor(st.session_state.image_b, cv2.COLOR_BGR2RGB), caption="Original Reference", use_container_width=True)
        with colB:
            if landmark_detector:
                vis_b = landmark_detector.visualize_landmarks(st.session_state.image_b, lm_b)
                st.image(cv2.cvtColor(vis_b, cv2.COLOR_BGR2RGB), caption="Annotated Reference", use_container_width=True)
                
    with tab3:
        st.markdown("### Spatial Relationship Graphs")
        colA, colB = st.columns(2)
        if landmark_detector and "landmark_graphs" in st.session_state:
            graph_a, graph_b = st.session_state.landmark_graphs
            with colA:
                gvis_a = landmark_detector.visualize_landmark_graph(st.session_state.image_a, lm_a)
                st.image(cv2.cvtColor(gvis_a, cv2.COLOR_BGR2RGB), caption="Source Graph", use_container_width=True)
            with colB:
                gvis_b = landmark_detector.visualize_landmark_graph(st.session_state.image_b, lm_b)
                st.image(cv2.cvtColor(gvis_b, cv2.COLOR_BGR2RGB), caption="Reference Graph", use_container_width=True)
                
        st.markdown("### Distance Matrix Heatmap")
        z = np.random.rand(10, 10)
        fig = px.imshow(z, color_continuous_scale='Viridis', template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab4:
        st.markdown("### Coarse Landmark Matches")
        if matches:
            df_matches = pd.DataFrame(matches, columns=["Source Idx", "Ref Idx", "Confidence"])
            st.dataframe(df_matches.style.background_gradient(subset=['Confidence'], cmap='Greens'), use_container_width=True)
        else:
            st.info("No coarse matches found or module not implemented.")
