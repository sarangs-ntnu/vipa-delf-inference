# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st
import random
from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision
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
import torch
import torchvision.transforms as transforms
from PIL import Image
import os


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
#image = st.file_uploader("Upload your image in JPG or PNG format", type=["jpg", "png"])

if image is not None:

    # Define device
    use_cuda=torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    device=torch.device("cuda:0" if use_cuda else "mps" if use_mps else "cpu")
    device = "cpu"

    #define transformation
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of DenseNet-121
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    test_model_path = "C_30_32_01.pth"
    n_classes = 2

    test_model=torchvision.models.densenet121(weights=False).to(device)

    n_inputs = test_model.classifier.in_features
    test_model.classifier = nn.Sequential(
              nn.Linear(n_inputs, n_classes),               
              nn.LogSoftmax(dim=1))


    checkpoint=torch.load(test_model_path,map_location=device)   # loading best model
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.to(device)
    test_model.eval()

    # Function to load and preprocess image
    def load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    input_image = load_image(image).to(device)

    # Perform inference
    with torch.no_grad():
        output = test_model(input_image)

        # Get the predicted class index
        _, predicted = torch.max(output, 1)
        predicted_class_index = predicted.item()

        # Interpret the results (assuming you have a class label mapping)
        class_labels = ['Disease','Healthy']  # Replace with your actual class labels
        predicted_class_label = class_labels[predicted_class_index]
        if predicted_class_label == 'Healthy':
            st.write(f"The input vineyard image leaf shows : :green[{predicted_class_label}] symptoms.")
        else:
            st.write(f"The input vineyard image leaf shows : :red[{predicted_class_label}] symptoms.")


    # LIME Implementation
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    def batch_predict(images):
        test_model.eval()
        batch = torch.stack([transform(Image.fromarray(i)) for i in images], dim=0)
        batch = batch.to(device)
        with torch.no_grad():
            logits = test_model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()


    image = Image.open(image)

    explanation = explainer.explain_instance(np.array(image), 
                                             batch_predict, 
                                             top_labels=2, 
                                             hide_color=0, 
                                             num_samples=50,
                                             segmentation_fn=None)

    # Map each explanation to the corresponding label
    temp, mask = explanation.get_image_and_mask(predicted_class_index, positive_only=True, num_features=10, hide_rest=False)
    lime_image = mark_boundaries(temp, mask, color=(1, 0, 0))
    lime_image_path = "lime_explanation.png"
    import matplotlib.pyplot as plt
    plt.imsave(lime_image_path, lime_image)

    # Display the uploaded image and LIME explanation in a grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader(f"LIME Explanation {predicted_class_label}")
        st.image(lime_image_path, use_column_width=True)
        st.write(f"Saved at: `{os.path.abspath(lime_image_path)}`")

    # Display the LIME explanation
    #st.image(lime_image, caption=f'LIME: {predicted_class_index}', use_column_width=True)

    # Optionally, display the original image and class label
    #st.image(image, caption='Original Image', use_column_width=True)
    #st.write(f'Predicted Class Index: {predicted_class_index}')