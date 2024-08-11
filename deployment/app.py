import streamlit as st

st.set_page_config(
    page_title="Adobe Gensolve Project",
    page_icon="👋",
)

st.write("Adobe Gensolve Project")

st.markdown(
    """
    ## Welcome to the Adobe Gensolve Project
    This is a multi-page web app that demonstrates different computer vision algorithms
    
    The app is divided into the following pages:
    
    1. **Algorithm 1:** Object completion using Harris Corner Detection and Symmetrical Analysis
    2. **Algorithm 2:** A page that demonstrates the Generalized Hough Transform algorithm
    3. **Algorithm 3:** A page that demonstrates the Generalized Hough Transform algorithm with multi-scale and multi-shift detection
    4. **Algorithm 4:** Interpolation using Area-based and Edge-based methods

"""
)