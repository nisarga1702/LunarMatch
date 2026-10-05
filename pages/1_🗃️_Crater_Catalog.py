import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from core.crater_catalog import (
    get_all_craters, get_crater_images, add_crater, add_crater_image,
    delete_crater, get_catalog_stats, get_all_albums
)
from core.catalog_builder import detect_and_number_craters, extract_crater_patch, populate_default_catalog

st.set_page_config(page_title="Reference Albums | LunarMatch", page_icon="🖼️", layout="wide")

# Ensure default catalog is populated
populate_default_catalog()

st.title("🖼️ Lunar Crater Reference Albums & Training Datasets")
st.caption("Browse Reference Albums, Trained Observation Images, and Database Feature Collections")

tab1, tab2, tab3 = st.tabs(["📚 Reference Albums Gallery", "➕ Add New Reference Image", "📊 Database Stats"])

# ── TAB 1: Reference Albums Gallery ──
with tab1:
    st.subheader("Reference Crater Albums & Training Collections")
    albums = get_all_albums()
    
    if not albums:
        st.info("No albums in database yet. Populate default catalog or add a reference image.")
    else:
        # Album Selection
        selected_id = st.session_state.get('selected_album_id', albums[0]['id'])
        
        col_select, col_info = st.columns([1, 2])
        with col_select:
            album_options = [a['id'] for a in albums]
            selected_album_id = st.selectbox(
                "Select Album to View:",
                options=album_options,
                index=album_options.index(selected_id) if selected_id in album_options else 0,
                format_func=lambda aid: next(f"🖼️ Album #{a['id']}: {a['name']} ({a['image_count']} photos)" for a in albums if a['id'] == aid)
            )
            
        target_album = next(a for a in albums if a['id'] == selected_album_id)
        images = get_crater_images(selected_album_id)
        
        st.markdown("---")
        # Album Header Card
        st.markdown(f"""
        <div class='glass-card' style='border-left: 5px solid #00b4d8;'>
            <h2>🖼️ Album #{target_album['id']}: {target_album['name']} ({target_album['label']})</h2>
            <p><b>Total Trained Reference Photos:</b> <span style='font-size:1.3rem; color:#00ff9d;'>{len(images)}</span> | 
            <b>Diameter:</b> {target_album['diameter_km']} km | 
            <b>Coordinates:</b> {target_album['latitude']}°, {target_album['longitude']}°</p>
            <p><b>Data Source:</b> {target_album['source']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.subheader(f"Photos in {target_album['name']} Album ({len(images)} images)")
        
        if images:
            img_cols = st.columns(min(4, max(1, len(images))))
            for idx, img_data in enumerate(images):
                with img_cols[idx % len(img_cols)]:
                    img_rgb = cv2.cvtColor(img_data['image'], cv2.COLOR_BGR2RGB)
                    is_trained = img_data['image_type'] == 'trained_reference'
                    badge_str = "🎓 Trained Observation" if is_trained else "📌 Base Reference"
                    
                    st.image(
                        img_rgb,
                        caption=f"Photo #{idx+1} | {badge_str}\nSun Az: {img_data['sun_azimuth']}° | Sensor: {img_data['sensor']}",
                        use_container_width=True
                    )
                    st.caption(f"Added: {img_data['added_at'][:10] if img_data.get('added_at') else 'N/A'}")
        else:
            st.warning("No photos currently in this album.")

# ── TAB 2: Add New Reference Image (Detect & Number) ──
with tab2:
    st.subheader("Upload Reference Image to Create/Expand Albums")
    st.write("Upload a lunar terrain image to detect craters, number them automatically (e.g., C-001, C-002), and extract patches into reference albums.")
    
    uploaded_file = st.file_uploader("Upload Lunar Image (PNG/JPG/TIF):", type=["png", "jpg", "jpeg", "tif"])
    
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
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
        
        if st.button("💾 Create Albums & Save Craters to Database"):
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
            st.success(f"Successfully created {added_count} reference crater albums in database!")
            st.rerun()

# ── TAB 3: Database Stats ──
with tab3:
    st.subheader("Database Overview")
    stats = get_catalog_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Catalog Albums", stats['total_craters'])
    with col2:
        st.metric("Total Stored Photos", stats['total_images'])
    with col3:
        st.metric("Confirmed Trained Samples", stats['confirmed_images'])
    
    st.subheader("Raw Album Records")
    all_cr = get_all_craters()
    if all_cr:
        df = pd.DataFrame(all_cr)
        st.dataframe(df, use_container_width=True)
