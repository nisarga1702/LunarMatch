import streamlit as st
import os
import cv2
import pandas as pd
import json

st.set_page_config(page_title="Results | LunarMatch", page_icon="📊", layout="wide")

css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Step 5: Results & Summary")

if "pipeline_results" not in st.session_state:
    st.warning("⚠️ No registration results found. Please run the Registration pipeline first.")
    st.stop()

res = st.session_state.pipeline_results
metrics = res.get("metrics", {})

st.markdown("## Executive Summary")
conf = metrics.get('confidence', 0.0)
if conf > 80:
    st.success(f"### Overall Confidence: {conf:.1f}% — STATUS: PASS ✅")
elif conf > 50:
    st.warning(f"### Overall Confidence: {conf:.1f}% — STATUS: MARGINAL ⚠️")
else:
    st.error(f"### Overall Confidence: {conf:.1f}% — STATUS: FAIL ❌")

st.markdown("---")

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### Metrics Summary")
    df_metrics = pd.DataFrame({
        "Metric": ["Inlier Count", "Inlier Ratio (%)", "RMSE (px)", "Spatial Coverage (%)", "Processing Time (s)"],
        "Value": [
            metrics.get("inlier_count", 0), 
            f"{metrics.get('inlier_ratio', 0.0):.1f}", 
            f"{metrics.get('rmse', 0.0):.2f}", 
            f"{metrics.get('coverage_pct', 0.0):.1f}", 
            f"{metrics.get('time_s', 0.0):.2f}"
        ]
    })
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)
    
    st.markdown("### Export")
    metrics_json = json.dumps(metrics, indent=4)
    st.download_button(
        label="Download Metrics (JSON)",
        data=metrics_json,
        file_name="lunarmatch_metrics.json",
        mime="application/json"
    )
    
    if "match_points" in res:
        pts = res["match_points"]
        df_pts = pd.DataFrame(pts, columns=["source_x", "source_y", "ref_x", "ref_y", "confidence"])
        csv = df_pts.to_csv(index=False)
        st.download_button(
            label="Download Match Points (CSV)",
            data=csv,
            file_name="lunarmatch_points.csv",
            mime="text/csv"
        )
        
with col2:
    st.markdown("### Match Points Sample")
    if "match_points" in res:
        st.dataframe(df_pts.head(100), use_container_width=True)
    else:
        st.info("No match points data available.")

st.markdown("---")

st.markdown("### Visual Summary")
vc1, vc2, vc3 = st.columns(3)
with vc1:
    st.markdown("**Source Image**")
    if "image_a" in st.session_state:
        st.image(cv2.cvtColor(st.session_state.image_a, cv2.COLOR_BGR2RGB), use_container_width=True)
with vc2:
    st.markdown("**Correspondences**")
    if "ransac_vis" in res:
        st.image(cv2.cvtColor(res["ransac_vis"], cv2.COLOR_BGR2RGB), use_container_width=True)
with vc3:
    st.markdown("**Registered Image**")
    if "warped_image" in res:
        st.image(cv2.cvtColor(res["warped_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

st.markdown("---")
st.markdown("### Pipeline Architecture")
st.info("✔️ Image Input ➔ ✔️ Landmark Detection ➔ ✔️ Sun Geometry Constraint ➔ ✔️ RANSAC Registration ➔ ✔️ Results")

st.markdown("### Innovation Highlights")
st.markdown("""
1. **Multi-modal capability**: Capable of handling OHRC to LROC matching.
2. **Structural Graph Matching**: Employs crater/ridge spatial relationships.
3. **Sun Geometry Constraints**: Filters false positives using illumination directions.
4. **Resilient Feature Pipeline**: Combines SIFT/AKAZE with spatial distribution scoring.
""")

st.markdown("### Next Steps (Phase 2 & 3)")
st.markdown("""
- **TMC-2 and IIRS Support**: Expanding to multi-resolution cross-modal matching.
- **Hardware Acceleration**: GPU optimization for real-time edge processing on spacecraft.
- **Deep Learning Embeddings**: Using transformer-based feature descriptors.
""")
