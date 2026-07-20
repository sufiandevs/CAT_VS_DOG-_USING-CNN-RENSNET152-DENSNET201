import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown # New import for downloading from Google Drive

# --- Configuration --- #
IMG_HEIGHT = 224 # Must match the height used during training
IMG_WIDTH = 224  # Must match the width used during training
class_names = ['cat', 'dog'] # Must match the class names used during training

# --- Google Drive File IDs for Models --- #
# IMPORTANT: Replace these with your actual Google Drive file IDs.
# Ensure these files are publicly shareable or accessible via a shared link.
# To get a file ID: Upload your .h5 model to Google Drive, right-click, 'Get link', and copy the ID part from the URL.
DRIVE_FILE_IDS = {
    'simple_cnn': '1eiaSHApxdYxrm8okQxWzgTOSx62mYTst', # e.g., '1aBcDeFgHiJkLmNoPqRsTuVwXyZ0123'
    'densenet': '1OCcUXJ0Kz6qB8Fy9GAFH6Rfm2DHBmGoN',
    'resnet': '11J20MhOtA9J1KfFT_6NZls9ShC-p6kLr3',
}

# Directory to temporarily store downloaded models
TEMP_MODEL_DIR = 'temp_models'
os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

# --- Load Models --- #
@st.cache_resource
def load_model_from_drive(model_name_key, local_filename):
    file_id = DRIVE_FILE_IDS.get(model_name_key)
    if not file_id or file_id.startswith('YOUR_'):
        st.error(f"Google Drive File ID for {model_name_key} is not configured or is a placeholder. Please update DRIVE_FILE_IDS.")
        return None

    local_path = os.path.join(TEMP_MODEL_DIR, local_filename)

    if not os.path.exists(local_path):
        st.info(f"Downloading {local_filename} from Google Drive. This may take a moment...")
        try:
            # Use gdown to download the file
            gdown.download(f'https://drive.google.com/uc?id={file_id}', local_path, quiet=True, fuzzy=True)
            st.success(f"Downloaded {local_filename}.")
        except Exception as e:
            st.error(f"Error downloading {local_filename} from Google Drive. Check the file ID and permissions: {e}")
            return None
    else:
        st.info(f"{local_filename} already exists locally. Skipping download.")

    try:
        model = tf.keras.models.load_model(local_path)
        return model
    except Exception as e:
        st.error(f"Error loading {local_filename} after download: {e}")
        return None

# Update loading functions to use the new load_model_from_drive helper
def load_simple_cnn_model():
    return load_model_from_drive('simple_cnn', 'simple_cnn_model.h5')

def load_densenet_model():
    return load_model_from_drive('densenet', 'densenet201_model.h5')

def load_resnet_model():
    return load_model_from_drive('resnet', 'resnet152_model.h5')

simple_cnn_model = load_simple_cnn_model()
densenet_model = load_densenet_model()
resnet_model = load_resnet_model()

# --- Preprocessing Functions (must match training preprocessing) --- #
def preprocess_simple_cnn(image_array):
    # Rescale pixel values from [0, 255] to [0, 1]
    return tf.cast(image_array, tf.float32) / 255.0

def preprocess_densenet(image_array):
    # DenseNet's preprocess_input expects pixel values in the range [0, 255]
    image_array = tf.cast(image_array, tf.float32)
    return tf.keras.applications.densenet.preprocess_input(image_array)

def preprocess_resnet(image_array):
    # ResNet's preprocess_input also expects pixel values in the range [0, 255]
    image_array = tf.cast(image_array, tf.float32)
    return tf.keras.applications.resnet.preprocess_input(image_array)

# --- Streamlit UI --- #
st.title('Cat vs Dog Classifier')
st.write('Upload an image of a cat or a dog to get a prediction using different CNN models.')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Resize image for models
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.asarray(image_resized)
    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)

    st.subheader('Select a Model for Prediction:')
    model_choice = st.radio(
        "",
        ('Simple CNN', 'DenseNet201', 'ResNet152'),
        index=0 # Default to Simple CNN
    )

    model_to_use = None
    preprocess_func = None
    model_name = ""

    if model_choice == 'Simple CNN':
        model_to_use = simple_cnn_model
        preprocess_func = preprocess_simple_cnn
        model_name = 'Simple CNN'
    elif model_choice == 'DenseNet201':
        model_to_use = densenet_model
        preprocess_func = preprocess_densenet
        model_name = 'DenseNet201'
    elif model_choice == 'ResNet152':
        model_to_use = resnet_model
        preprocess_func = preprocess_resnet
        model_name = 'ResNet152'

    if model_to_use:
        if preprocess_func:
            processed_image_batch = preprocess_func(image_batch)
            prediction = model_to_use.predict(processed_image_batch)
            predicted_class_index = (prediction > 0.5).astype(int)[0][0]
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][0] if predicted_class_index == 1 else (1 - prediction[0][0])

            st.write(f'### Prediction using {model_name}:')
            st.success(f'The image is a **{predicted_class.upper()}** with {confidence:.2%} confidence.')
        else:
            st.warning("Preprocessing function not defined for the selected model.")
    else:
        st.warning(f"The {model_choice} model could not be loaded.")
else:
    st.info("Please upload an image to make a prediction.")
