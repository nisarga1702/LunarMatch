import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from core.crater_catalog import (
    get_all_craters, get_crater_images, add_crater, add_crater_image,
    delete_crater, get_catalog_stats
)
from core.catalog_builder import detect_and_number_craters, extract_crater_patch, populate_default_catalog

st.set_page_config(page_title="Crater Catalog | LunarMatch", page_icon="🗃️", layout="wide")

# Ensure default catalog is populated
populate_default_catalog()

st.title("🗃️ Lunar Crater Catalog & Database")
st.caption("Reference Database of Numbered Lunar Craters (LROC NAC / ISRO)")

tab1, tab2, tab3 = st.tabs(["📜 Explore Catalog", "➕ Add New Reference Image", "📊 Database Stats"])

# ── TAB 1: Explore Existing Catalog ──
with tab1:
    st.subheader("Reference Craters in Database")
    craters = get_all_craters()
    
    if not craters:
        st.info("No craters in catalog yet. Populate defaults or upload a reference image.")
    else:
        # Display as grid / selectbox
        col_sel, col_empty = st.columns([1, 2])
        with col_sel:
            selected_crater_id = st.selectbox(
                "Select a Crater to Inspect:",
                options=[c['id'] for c in craters],
                format_func=lambda cid: next(f"[{c['label']}] {c['name']} ({c['diameter_km']} km)" for c in craters if c['id'] == cid)
            )
        
        target_crater = next(c for c in craters if c['id'] == selected_crater_id)
        images = get_crater_images(selected_crater_id)
        
        st.markdown("---")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Crater Name", target_crater['name'])
        with m_col2:
            st.metric("Label ID", target_crater['label'])
        with m_col3:
            st.metric("Diameter", f"{target_crater['diameter_km']} km")
        with m_col4:
            st.metric("Coordinates", f"{target_crater['latitude']}°, {target_crater['longitude']}°")
            
        st.write(f"**Data Source:** {target_crater['source']}")
        
        # Display reference images for this crater
        st.subheader("Reference & Verified Linked Images")
        if images:
            img_cols = st.columns(min(4, len(images)))
            for idx, img_data in enumerate(images):
                with img_cols[idx % len(img_cols)]:
                    img_rgb = cv2.cvtColor(img_data['image'], cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption=f"Type: {img_data['image_type']} | Sun Az: {img_data['sun_azimuth']}°", use_container_width=True)
                    st.caption(f"Sensor: {img_data['sensor']} | Added: {img_data['added_at'][:10] if img_data.get('added_at') else 'N/A'}")
        else:
            st.warning("No reference images stored for this crater.")

# ── TAB 2: Add New Reference Image (Detect & Number) ──
with tab2:
    st.subheader("Upload Reference Image to Flag & Number Craters")
    st.write("Upload a lunar terrain image to detect craters, number them automatically (e.g., C-001, C-002), and save them to the reference dataset.")
    
    uploaded_file = st.file_uploader("Upload Lunar Image (PNG/JPG/TIF):", type=["png", "jpg", "jpeg", "tif"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=uint8) if 'uint8' in globals() else np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Reference Image", use_container_width=True)
        
        if st.button("🔍 Detect & Number Craters"):
            with st.spinner("Detecting lunar craters..."):
                detections = detect_and_number_craters(img_gray)
                
                if not detections:
                    st.error("No craters detected in image. Try an image with higher crater contrast.")
                else:
                    st.session_state['detected_craters'] = detections
                    st.session_state['ref_image_bgr'] = img_bgr
                    st.session_state['ref_image_gray'] = img_gray
                    st.success(f"Detected {len(detections)} craters!")
                    
    if 'detected_craters' in st.session_state and 'ref_image_bgr' in st.session_state:
        st.subheader("Numbered Crater Map")
        img_vis = st.session_state['ref_image_bgr'].copy()
        
        for det in st.session_state['detected_craters']:
            cx, cy, r, label = det['center_x'], det['center_y'], det['radius_px'], det['label']
            cv2.circle(img_vis, (cx, cy), r, (0, 255, 200), 2)
            cv2.putText(img_vis, label, (cx - 15, cy - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
            
        st.image(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB), caption="Detected Craters Map", use_container_width=True)
        
        # Save to Database Button
        if st.button("💾 Save Numbered Craters to Catalog Database"):
            added_count = 0
            for det in st.session_state['detected_craters']:
                patch = extract_crater_patch(st.session_state['ref_image_bgr'], det['center_x'], det['center_y'], det['radius_px'])
                if patch is not None:
                    cid = add_crater(
                        name=f"Crater {det['label']}",
                        label=det['label'],
                        radius_px=det['radius_px'],
                        center_x=det['center_x'],
                        center_y=det['center_y'],
                        source='User Upload'
                    )
                    add_crater_image(cid, patch, image_type='reference', confirmed=True)
                    added_count += 1
            st.success(f"Successfully added {added_count} craters to the catalog database!")
            st.rerun()

# ── TAB 3: Database Stats ──
with tab3:
    st.subheader("Database Overview")
    stats = get_catalog_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Catalog Craters", stats['total_craters'])
    with col2:
        st.metric("Total Reference Images", stats['total_images'])
    with col3:
        st.metric("Confirmed Matches", stats['confirmed_images'])
    
    st.subheader("Raw Crater Records")
    all_cr = get_all_craters()
    if all_cr:
        df = pd.DataFrame(all_cr)
        st.dataframe(df, use_container_width=True)
