import streamlit as st
import cv2
import numpy as np
from core.crater_catalog import add_crater_image, get_crater, get_crater_images, get_catalog_stats
from core.catalog_builder import populate_default_catalog

st.set_page_config(page_title="Confirm & Update | LunarMatch", page_icon="✅", layout="wide")
populate_default_catalog()

st.title("✅ Confirm Match & Update Crater Reference Database")
st.caption("Human-in-the-loop Verification & Database Linked Entry")

if 'top_match' not in st.session_state or st.session_state['top_match'] is None:
    st.warning("⚠️ No active match candidate to confirm. Please run matching on Page 2 first.")
    st.stop()

top = st.session_state['top_match']
new_img = st.session_state.get('new_image_bgr')

crater_info = get_crater(top['crater_id'])

st.markdown("""
<div class='glass-card'>
    <h3>Review Candidate Match</h3>
    <p>Verify if the newly captured Chandrayaan-2 image matches the reference catalog record.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Side-by-side comparison
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 New Observation Image")
    st.caption("Chandrayaan-2 Capture")
    if new_img is not None:
        st.image(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB), caption="New Image", use_container_width=True)

with col2:
    st.subheader(f"🗃️ Catalog Reference: {top['crater_name']}")
    st.caption(f"Label: {top['label']} | Diameter: {crater_info['diameter_km'] if crater_info else 'N/A'} km")
    if top['ref_image'] is not None:
        st.image(cv2.cvtColor(top['ref_image'], cv2.COLOR_BGR2RGB), caption="Catalog Reference Patch", use_container_width=True)

st.write("---")

# Feature Matching Overlay
st.subheader("Keypoint Correspondence Alignment")
if top['vis_image'] is not None:
    st.image(cv2.cvtColor(top['vis_image'], cv2.COLOR_BGR2RGB), caption="Verified Feature Correspondences", use_container_width=True)

# Metrics Summary Card
st.markdown(f"""
<div class='glass-card' style='text-align: center; border-color: #00ff9d;'>
    <h2>Identified as: <span class='gradient-text'>{top['crater_name']} ({top['label']})</span></h2>
    <h3>Confidence Score: <span style='color: #00ff9d;'>{top['confidence']}%</span></h3>
    <p>Inliers: <b>{top['inliers']}</b> | Raw Feature Matches: <b>{top['raw_matches']}</b></p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Interactive Confirmation Buttons
col_btn1, col_btn2, col_space = st.columns([2, 2, 3])

with col_btn1:
    if st.button("✅ Confirm Match & Save to Reference Database (Click OK)", type="primary", use_container_width=True):
        if new_img is not None:
            # Add to database
            add_crater_image(
                crater_id=top['crater_id'],
                image=new_img,
                image_type='matched_observation',
                sensor='Chandrayaan-2 OHRC',
                confirmed=True
            )
            st.session_state['match_confirmed'] = True
            st.success(f"🎉 SUCCESS! The image has been confirmed and permanently linked to **{top['crater_name']} ({top['label']})** in the SQLite Reference Database!")
            st.rerun()

with col_btn2:
    if st.button("❌ Reject Match", use_container_width=True):
        st.session_state['top_match'] = None
        st.warning("Match candidate rejected. You can return to Page 2 to re-evaluate.")
        st.rerun()

st.write("---")

# Show linked images count for this crater
linked_imgs = get_crater_images(top['crater_id'])
st.subheader(f"Existing Reference & Linked Images for {top['crater_name']} ({len(linked_imgs)} total)")
if linked_imgs:
    cols = st.columns(min(4, len(linked_imgs)))
    for idx, item in enumerate(linked_imgs):
        with cols[idx % len(cols)]:
            st.image(cv2.cvtColor(item['image'], cv2.COLOR_BGR2RGB), caption=f"Type: {item['image_type']}\nAdded: {item['added_at'][:10] if item.get('added_at') else 'N/A'}", use_container_width=True)
