# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st
import random
from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision import models
from lime import lime_image
from skimage.segmentation import mark_boundaries
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10


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
        st.title("VIPA-DELF")
        st.subheader("Vineyard leaf disease prediction is a crucial aspect of viticulture aimed at identifying and managing diseases that can affect the health and productivity of grapevines. Effective prediction allows vineyard managers to take timely actions to mitigate the spread of diseases, ensuring a healthy yield.")
        st.write("The 'VIPA-DELF' sub-project has indirectly received funding from the European Union's Horizon Europe research and innovation action programme, via the CHAMELEON Open Call #1 issued and executed under the CHAMELEON project (Grant Agreement no. 101060529)")

st.write("""
         # Vineyard Leaf Disease Prediction
         """
         )


#get input image
image = st.camera_input("Take a picture")
if image is not None:
	st.image(image, caption='Captured Image')

# Define device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
st.write(device)