# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st
import random
from PIL import Image, ImageOps
import numpy as np


# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Vineyard Leaf Disease Prediction",
    page_icon = ":leaf:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

with st.sidebar:
        #st.image('mg.png')
        st.title("Mangifera Healthika")
        st.subheader("Vineyard leaf disease prediction is a crucial aspect of viticulture aimed at identifying and managing diseases that can affect the health and productivity of grapevines. Effective prediction allows vineyard managers to take timely actions to mitigate the spread of diseases, ensuring a healthy yield.")

st.write("""
         # Vineyard Leaf Disease Prediction
         """
         )


#get input image
image = st.camera_input("Take a picture")
if image is not None:
	st.image(image, caption='Captured Image')