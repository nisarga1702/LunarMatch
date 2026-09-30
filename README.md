# LunarMatch–TGS Prototype
A lightweight SIH demo implementing the proposed registration workflow.

## Demonstrates
- OHRC / TMC-2 / IIRS selection
- LRO / SELENE reference selection
- Acquisition and Sun-angle metadata
- SIFT correspondence
- RANSAC outlier rejection
- RMSE, inlier ratio, spatial coverage and confidence
- Registered output
- Built-in demo images

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For real data, disable the demo checkbox and upload a Chandrayaan-2 image plus a corresponding LRO/SELENE image.

This is a prototype baseline. A full research implementation can add SPICE-derived Sun/orbit geometry, shadow-aware descriptors, sensor-specific preprocessing, learned cross-modal matching and true sub-pixel refinement.
