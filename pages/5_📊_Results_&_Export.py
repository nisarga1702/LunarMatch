import streamlit as st
import json
import cv2
import numpy as np
import pandas as pd
from core.crater_catalog import get_all_craters, get_crater_images, get_catalog_stats
from core.catalog_builder import populate_default_catalog

st.set_page_config(page_title="Results & Export | LunarMatch", page_icon="📊", layout="wide")
populate_default_catalog()

st.title("📊 Results & Catalog Analytics")
st.caption("Executive Summary & Data Export")

stats = get_catalog_stats()

# Overview Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Catalog Craters", stats['total_craters'])
with col2:
    st.metric("Total Stored Images", stats['total_images'])
with col3:
    st.metric("Confirmed Matches", stats['confirmed_images'])
with col4:
    st.metric("Pending Matches", stats['pending_images'])

st.write("---")

# Active Match Session Summary
st.subheader("Active Session Match Report")

if 'top_match' in st.session_state and st.session_state['top_match']:
    top = st.session_state['top_match']
    st.markdown(f"""
    <div class='glass-card' style='border-left: 5px solid #00ff9d;'>
        <h3>Latest Match Identification</h3>
        <p><b>Identified Crater:</b> {top['crater_name']} ({top['label']})</p>
        <p><b>Confidence Score:</b> {top['confidence']}%</p>
        <p><b>Inliers:</b> {top['inliers']} / {top['raw_matches']}</p>
        <p><b>Status:</b> {'✅ CONFIRMED & SAVED TO CATALOG' if st.session_state.get('match_confirmed') else '⏳ PENDING USER CONFIRMATION'}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No active match evaluated in this session. Run Page 2 (Match New Image) to view live match report.")

st.write("---")

# Database Records Table
st.subheader("Catalog Craters Summary Table")
craters = get_all_craters()
if craters:
    records = []
    for c in craters:
        imgs = get_crater_images(c['id'])
        records.append({
            "ID": c['id'],
            "Label": c['label'],
            "Name": c['name'],
            "Diameter (km)": c['diameter_km'],
            "Latitude": c['latitude'],
            "Longitude": c['longitude'],
            "Source": c['source'],
            "Stored Images": len(imgs)
        })
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True)

st.write("---")

# Safe JSON & CSV Export
st.subheader("📥 Export Pipeline & Catalog Data")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.write("**Export Catalog Summary (JSON):**")
    export_dict = {
        "catalog_stats": stats,
        "craters": [
            {
                "id": int(c['id']),
                "label": str(c['label']),
                "name": str(c['name']),
                "diameter_km": float(c['diameter_km']),
                "latitude": float(c['latitude']),
                "longitude": float(c['longitude']),
                "source": str(c['source'])
            }
            for c in craters
        ]
    }
    
    # Custom encoder for extra safety against any remaining numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
            
    json_str = json.dumps(export_dict, indent=4, cls=NumpyEncoder)
    st.download_button(
        label="💾 Download Catalog JSON Report",
        data=json_str,
        file_name="lunar_crater_catalog_report.json",
        mime="application/json"
    )

with col_exp2:
    st.write("**Export Catalog Table (CSV):**")
    if craters:
        csv_str = df.to_csv(index=False)
        st.download_button(
            label="💾 Download Catalog CSV Table",
            data=csv_str,
            file_name="lunar_crater_catalog.csv",
            mime="text/csv"
        )
