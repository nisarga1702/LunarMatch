import streamlit as st
import os

# Page config
st.set_page_config(
    page_title="LunarMatch-TGS | SIH 2026",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("Custom CSS not found.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 class='gradient-text'>LunarMatch-TGS</h2>", unsafe_allow_html=True)
    st.caption("Team Name | SIH 2026")
    st.markdown("---")
    st.markdown("👈 **Use the sidebar above to navigate through the pipeline stages.**")

# --- Main Page ---
# Hero Section
st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'><span class='gradient-text'>🌙 LunarMatch-TGS</span></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: var(--text-muted);'>Temporal + Geometric + Shadow Guided Lunar Image Registration</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><span class='tech-tag'>SIH 2026</span> <span class='tech-tag'>Problem Statement 26166</span> <span class='tech-tag'>ISRO</span></p>", unsafe_allow_html=True)

st.write("---")

# Problem Overview
st.markdown("""
<div class='glass-card'>
    <h3 class='gradient-text'>🚀 Mission Overview</h3>
    <p>Finding accurate correspondences between Chandrayaan-2 images (OHRC/TMC/IIRS) and reference imagery (LROC NAC) is challenging due to extreme differences in sun angles, scale, and orbital viewpoints. LunarMatch-TGS introduces a robust pipeline that incorporates geometry, illumination, and structural features to achieve highly confident registrations.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Key Innovations
st.markdown("<h3 class='gradient-text'>✨ Key Innovations</h3>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='glass-card' style='height: 100%;'>
        <h4>🌑 Landmark-Guided</h4>
        <p style='font-size: 0.9em; color: var(--text-muted);'>Detect craters, ridges & terrain structures for coarse-to-fine alignment.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='glass-card' style='height: 100%;'>
        <h4>☀️ Sun Geometry</h4>
        <p style='font-size: 0.9em; color: var(--text-muted);'>Use acquisition time → Sun position → illumination/shadow consistency.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='glass-card' style='height: 100%;'>
        <h4>🔗 Multi-Stage</h4>
        <p style='font-size: 0.9em; color: var(--text-muted);'>SIFT + RANSAC + spatial distribution + sub-pixel refinement.</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='glass-card' style='height: 100%;'>
        <h4>📊 Confidence Scoring</h4>
        <p style='font-size: 0.9em; color: var(--text-muted);'>Structure + Geometry + Illumination = Correspondence Confidence.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# Pipeline Architecture
st.markdown("<h3 class='gradient-text'>⚙️ Pipeline Architecture</h3>", unsafe_allow_html=True)
st.markdown("""
<div class='pipeline-container'>
    <div class='pipeline-node'>CH2 Image</div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-node'>Pre-processing</div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-node'>Landmark Detection</div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-node'>Sun Geometry</div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-node'>Coarse Matching</div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-node'>Local Feature Matching</div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-node'>Confidence Filtering</div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-node'>RANSAC</div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-node'>Spatial Dist.</div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-node'>Sub-pixel Refinement</div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-node'>Registered Output</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# Tech Stack
st.markdown("""
<div class='glass-card' style='text-align: center;'>
    <h4>Powered By</h4>
    <div>
        <span class='tech-tag'>Python</span>
        <span class='tech-tag'>Streamlit</span>
        <span class='tech-tag'>OpenCV</span>
        <span class='tech-tag'>SIFT/AKAZE</span>
        <span class='tech-tag'>NumPy</span>
        <span class='tech-tag'>SciPy</span>
    </div>
</div>
""", unsafe_allow_html=True)
