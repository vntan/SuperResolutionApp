import streamlit as st
from ui import upscaler_ui, enhancer_ui

st.set_page_config(layout="wide")

st.header('SUPER RESOLUTION')
# st.title("Image Upscaler and Enhancer")
# tab1, tab2 = st.tabs([  "Upscaler", "Enhancer"]) #

# with tab1:
#     upscaler_ui.ui()

upscaler_ui.ui()

# with tab2:
#     enhancer_ui.ui()

