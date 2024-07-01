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
#image = st.camera_input("Take a picture")
#if image is not None:
	#st.image(image, caption='Captured Image')
image = st.file_uploader(
    "Upload your image in JPG or PNG format", type=["jpg", "png"]
)

if image is not None:

    # Define device
    use_cuda=torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    device=torch.device("cuda:0" if use_cuda else "mps" if use_mps else "cpu")
    device = "cpu" 
    device

    model_path = 'C_30_32.pth'

    #Initialize model
    print('Best model path:{}'.format(model_path))
    denseNet_model= models.densenet121(weights=False).to(device)

    n_classes = 2

    n_inputs = denseNet_model.classifier.in_features
    denseNet_model.classifier = nn.Sequential(
              nn.Linear(n_inputs, n_classes),               
              nn.LogSoftmax(dim=1))

    checkpoint=torch.load(model_path,map_location=device)   # loading best model
    # change name of model dictionary key as per your model key defined while saving the model.
    denseNet_model.load_state_dict(checkpoint['model_state_dict'])
    denseNet_model.to(device)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4762, 0.3054, 0.2368], std=[0.3345, 0.2407, 0.2164])
    ])

    image = Image.open(image)

    image = Image.open(image_path)

    # Apply the transform to the image
    image_tensor  = transform(image)

    # Add batch dimension
    image_tensor  = image_tensor.unsqueeze(0)
    image_tensor  = image_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output = denseNet_model(image_tensor)

        # Get the predicted class index
        _, predicted = torch.max(output, 1)
        predicted_class_index = predicted.item()
        
        # Interpret the results (assuming you have a class label mapping)
        class_labels = ['Apple___healthy', 'Apple___Cedar_apple_rust', 'Apple___Black_rot', 'Apple___Apple_scab']  # Replace with your actual class labels
        predicted_class_label = class_labels[predicted_class_index]

    print('Predicted class:', predicted_class_label)

    # GradCAM Implementation
    def generate_gradcam_heatmap(model, image_tensor, target_layer):
        # Hook for the gradients
        gradients = []
        
        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # Register hook on the target layer
        hook = target_layer.register_backward_hook(backward_hook)

        # Forward pass
        model.eval()
        output = denseNet_model(image_tensor)
        pred_class = output.argmax(dim=1).item()
        
        # Zero the gradients
        model.zero_grad()
        
        # Backward pass
        class_loss = output[0, pred_class]
        class_loss.backward()

        # Get the gradients and the activations
        gradients = gradients[0].cpu().data.numpy()[0]
        activations = target_layer.weight.data.cpu().numpy()[0]
        
        # Calculate the weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Calculate the GradCAM heatmap
        gradcam_heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            gradcam_heatmap += w * activations[i]
            
        # Apply ReLU and normalize
        gradcam_heatmap = np.maximum(gradcam_heatmap, 0)
        gradcam_heatmap = gradcam_heatmap / gradcam_heatmap.max()
        
        # Resize the heatmap to match the input image
        gradcam_heatmap = np.uint8(255 * gradcam_heatmap)
        gradcam_heatmap = Image.fromarray(gradcam_heatmap).resize((image.size[0], image.size[1]), Image.ANTIALIAS)
        gradcam_heatmap = np.asarray(gradcam_heatmap)
        
        # Remove the hook
        hook.remove()
        
        return gradcam_heatmap

    # Get the last convolutional layer
    target_layer = denseNet_model.features.denseblock4.denselayer16.conv2

    # Generate GradCAM heatmap
    gradcam_heatmap = generate_gradcam_heatmap(denseNet_model, image_tensor, target_layer)

    # Overlay the heatmap on the original image
    plt.imshow(image)
    plt.imshow(gradcam_heatmap, cmap='jet', alpha=0.5)
    plt.title(f'GradCAM: {predicted_class_label}')
    plt.axis('off')
    plt.show()

    # LIME Implementation
    def batch_predict(images):
        denseNet_model.eval()
        batch = torch.stack([transform(Image.fromarray(i)) for i in images], dim=0)
        batch = batch.to(device)
        logits = denseNet_model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(np.array(image), 
                                             batch_predict, 
                                             top_labels=2, 
                                             hide_color=0, 
                                             num_samples=1000,
                                             segmentation_fn=None)

    # Map each explanation to the corresponding label
    temp, mask = explanation.get_image_and_mask(predicted_class_index, positive_only=True, num_features=10, hide_rest=False)
    lime_image = mark_boundaries(temp, mask, color=(1, 0, 0))

    # Display LIME explanation
    plt.imshow(lime_image)
    plt.title(f'LIME: {predicted_class_label}')
    plt.axis('off')
    plt.show()