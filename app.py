# === FIX: Must be before ANY tensorflow import ===
import sys

def _patch_keras_3_to_2():
    """Strip Keras 3.x fields that break Keras 2.x loading"""
    try:
        # Patch keras.src.layers.layer.Layer (Keras 3 internal path)
        from keras.src.layers.layer import Layer
        _orig_from_config = Layer.from_config
        
        @classmethod
        def _fixed_from_config(cls, config):
            if isinstance(config, dict):
                config.pop('quantization_config', None)
                config.pop('lora_rank', None)
                # Also fix dtype if it's a dict
                dtype = config.get('dtype')
                if isinstance(dtype, dict) and dtype.get('class_name') == 'DTypePolicy':
                    config['dtype'] = dtype.get('config', {}).get('name', 'float32')
            return _orig_from_config(config)
        
        Layer.from_config = _fixed_from_config
        print("✅ Patch applied: keras.src.layers.layer.Layer")
    except Exception as e:
        print(f"Patch attempt 1: {e}")
    
    try:
        # Also patch tf.keras.layers.Layer as fallback
        import tensorflow as tf
        _orig_tf_from_config = tf.keras.layers.Layer.from_config
        
        @classmethod
        def _fixed_tf_from_config(cls, config):
            if isinstance(config, dict):
                config.pop('quantization_config', None)
                config.pop('lora_rank', None)
                dtype = config.get('dtype')
                if isinstance(dtype, dict) and dtype.get('class_name') == 'DTypePolicy':
                    config['dtype'] = dtype.get('config', {}).get('name', 'float32')
            return _orig_tf_from_config(config)
        
        tf.keras.layers.Layer.from_config = _fixed_tf_from_config
        print("✅ Patch applied: tf.keras.layers.Layer")
    except Exception as e:
        print(f"Patch attempt 2: {e}")

_patch_keras_3_to_2()
# === END FIX ===
# ... rest of your code exactly as before
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# --- Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
class_names = ['cat', 'dog']

# --- Google Drive File IDs (PASTE YOUR NEW .h5 IDs HERE) ---
DRIVE_FILE_IDS = {
    'simple_cnn': '18e6SqmW0ivF9wzKvp-vHH7v60dkFc1Pf',
    'densenet': '1J2o60_4nPvlhdN5H5yRKTGdfiBhcEisp',
    'resnet': '18gFrRbQA4NYngKrJtmprA3Dn7TOkmjdj',
}

TEMP_MODEL_DIR = 'temp_models'
os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

# --- Load Models ---
@st.cache_resource
def load_model_from_drive(model_name_key, local_filename):
    file_id = DRIVE_FILE_IDS.get(model_name_key)
    if not file_id or file_id.startswith('PASTE'):
        st.error(f"Google Drive File ID for {model_name_key} is not configured.")
        return None

    local_path = os.path.join(TEMP_MODEL_DIR, local_filename)

    if not os.path.exists(local_path):
        st.info(f"Downloading {local_filename}...")
        try:
            gdown.download(f'https://drive.google.com/uc?id={file_id}', local_path, quiet=True)
            st.success(f"Downloaded {local_filename}.")
        except Exception as e:
            st.error(f"Error downloading {local_filename}: {e}")
            return None

    try:
        model = tf.keras.models.load_model(local_path)
        return model
    except Exception as e:
        st.error(f"Error loading {local_filename}: {e}")
        return None

def load_simple_cnn_model():
    return load_model_from_drive('simple_cnn', 'simple_cnn_model.h5')

def load_densenet_model():
    return load_model_from_drive('densenet', 'densenet201_model.h5')

def load_resnet_model():
    return load_model_from_drive('resnet', 'resnet152_model.h5')

simple_cnn_model = load_simple_cnn_model()
densenet_model = load_densenet_model()
resnet_model = load_resnet_model()

# --- Preprocessing ---
def preprocess_simple_cnn(image_array):
    return tf.cast(image_array, tf.float32) / 255.0

def preprocess_densenet(image_array):
    image_array = tf.cast(image_array, tf.float32)
    return tf.keras.applications.densenet.preprocess_input(image_array)

def preprocess_resnet(image_array):
    image_array = tf.cast(image_array, tf.float32)
    return tf.keras.applications.resnet.preprocess_input(image_array)

# --- Streamlit UI ---
st.title('Cat vs Dog Classifier')
st.write('Upload an image of a cat or a dog.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")

    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.asarray(image_resized)
    image_batch = np.expand_dims(image_array, axis=0)

    st.subheader('Select a Model:')
    model_choice = st.radio("", ('Simple CNN', 'DenseNet201', 'ResNet152'), index=0)

    model_to_use = None
    preprocess_func = None
    model_name = ""

    if model_choice == 'Simple CNN':
        model_to_use, preprocess_func, model_name = simple_cnn_model, preprocess_simple_cnn, 'Simple CNN'
    elif model_choice == 'DenseNet201':
        model_to_use, preprocess_func, model_name = densenet_model, preprocess_densenet, 'DenseNet201'
    elif model_choice == 'ResNet152':
        model_to_use, preprocess_func, model_name = resnet_model, preprocess_resnet, 'ResNet152'

    if model_to_use and preprocess_func:
        processed_image_batch = preprocess_func(image_batch)
        prediction = model_to_use.predict(processed_image_batch)
        predicted_class_index = (prediction > 0.5).astype(int)[0][0]
        predicted_class = class_names[predicted_class_index]
        confidence = prediction[0][0] if predicted_class_index == 1 else (1 - prediction[0][0])

        st.write(f'### Prediction using {model_name}:')
        st.success(f'The image is a **{predicted_class.upper()}** with {confidence:.2%} confidence.')
    else:
        st.warning(f"The {model_choice} model could not be loaded.")
else:
    st.info("Please upload an image.")
