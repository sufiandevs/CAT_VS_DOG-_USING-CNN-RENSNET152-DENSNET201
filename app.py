import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet201, ResNet152
from PIL import Image
import numpy as np
import os
import gdown

# --- Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
class_names = ['cat', 'dog']

# --- Google Drive File IDs (PASTE YOUR .weights.h5 IDs HERE) ---
DRIVE_FILE_IDS = {
    'simple_cnn': '1g0yR1gNp6ru_YeJSXjOlW2n5H2gouM9p',
    'densenet': '1w0pXzFVALNRPb3Up79soYJ0EKddaV60S',
    'resnet': '1QlLMsz4hIpiu5T4Tu3CZCaPG_yncfTAy',
}

TEMP_MODEL_DIR = 'temp_models'
os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

# --- Build Models from Scratch (No loading .keras or .h5 models) ---

def build_simple_cnn():
    """Rebuild your Simple CNN architecture exactly as trained"""
    model = tf.keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_densenet():
    """Rebuild DenseNet201 architecture exactly as trained"""
    base_model = DenseNet201(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def build_resnet():
    """Rebuild ResNet152 architecture exactly as trained"""
    base_model = ResNet152(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# --- Download and Load Weights ---

@st.cache_resource
def load_model_weights(model_name_key, build_func, weights_filename):
    file_id = DRIVE_FILE_IDS.get(model_name_key)
    if not file_id or file_id.startswith('PASTE'):
        st.error(f"Google Drive File ID for {model_name_key} is not configured.")
        return None

    weights_path = os.path.join(TEMP_MODEL_DIR, weights_filename)

    if not os.path.exists(weights_path):
        st.info(f"Downloading {weights_filename}...")
        try:
            gdown.download(f'https://drive.google.com/uc?id={file_id}', weights_path, quiet=True)
            st.success(f"Downloaded {weights_filename}.")
        except Exception as e:
            st.error(f"Error downloading {weights_filename}: {e}")
            return None

    try:
        # Build fresh model
        model = build_func()
        # Load weights
        model.load_weights(weights_path)
        # Compile (needed for predict)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading {weights_filename}: {e}")
        return None

def load_simple_cnn_model():
    return load_model_weights('simple_cnn', build_simple_cnn, 'simple_cnn_weights.weights.h5')

def load_densenet_model():
    return load_model_weights('densenet', build_densenet, 'densenet201_weights.weights.h5')

def load_resnet_model():
    return load_model_weights('resnet', build_resnet, 'resnet152_weights.weights.h5')

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
