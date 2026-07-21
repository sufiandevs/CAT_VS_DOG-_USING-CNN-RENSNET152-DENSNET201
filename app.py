import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration --- #
IMG_HEIGHT = 224
IMG_WIDTH = 224
class_names = ['cat', 'dog']

# --- Google Drive File IDs ---
DRIVE_FILE_IDS = {
    'simple_cnn_model_pkl': '14sMJtuVr3TNBOMOO7QZA08rj1HnDgRKe',
    'densenet_model_pkl': '17HI_5U2X0pYlIwp7zjIvCprugN00gAR2',
    'resnet_model_pkl': '1IYzFu_g3eF7l-B9-e8yyCHGEgnijjD78',
    'model_comparison_csv': '1kvvJKXNJ76JOkiKGM4GGLKHqYiWkXm6i',
    'training_histories_pkl': '1eXmns8MI3syNHjND1zcnn8UzuNPXrHaK',
    'confusion_matrices_data_pkl': '13syGZjwL92vqQ64uMYgYR2eItKz7X-80',
}

TEMP_DIR = 'streamlit_temp_data'
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Helper function to download from Google Drive --- #
@st.cache_resource
def download_from_drive(file_key, local_filename):
    file_id = DRIVE_FILE_IDS.get(file_key)
    if not file_id or file_id.startswith('YOUR_'):
        st.error(f"Google Drive File ID for {file_key} is not configured. Please update DRIVE_FILE_IDS.")
        return None

    local_path = os.path.join(TEMP_DIR, local_filename)

    if not os.path.exists(local_path):
        try:
            gdown.download(f'https://drive.google.com/uc?id={file_id}', local_path, quiet=True)
        except Exception as e:
            st.error(f"Error downloading {local_filename}: {e}")
            return None
    return local_path

# --- Load Models --- #
def load_pkl_model(model_key, filename):
    local_path = download_from_drive(model_key, filename)
    if local_path:
        try:
            return joblib.load(local_path)
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
    return None

simple_cnn_model = load_pkl_model('simple_cnn_model_pkl', 'simple_cnn_model.pkl')
densenet_model = load_pkl_model('densenet_model_pkl', 'densenet201_model.pkl')
resnet_model = load_pkl_model('resnet_model_pkl', 'resnet152_model.pkl')

# --- Load Evaluation Artifacts --- #
@st.cache_data
def load_results_df():
    local_path = download_from_drive('model_comparison_csv', 'model_comparison.csv')
    if local_path:
        try:
            return pd.read_csv(local_path)
        except Exception as e:
            st.error(f"Error loading model_comparison.csv: {e}")
    return None

@st.cache_data
def load_histories():
    local_path = download_from_drive('training_histories_pkl', 'training_histories.pkl')
    if local_path:
        try:
            return joblib.load(local_path)
        except Exception as e:
            st.error(f"Error loading training_histories.pkl: {e}")
    return None

@st.cache_data
def load_confusion_matrices_data():
    local_path = download_from_drive('confusion_matrices_data_pkl', 'confusion_matrices_data.pkl')
    if local_path:
        try:
            return joblib.load(local_path)
        except Exception as e:
            st.error(f"Error loading confusion_matrices_data.pkl: {e}")
    return None

results_df = load_results_df()
training_histories = load_histories()
confusion_matrices_data = load_confusion_matrices_data()

# --- Preprocessing Functions --- #
def preprocess_simple_cnn(image_array):
    return tf.cast(image_array, tf.float32) / 255.0

def preprocess_densenet(image_array):
    image_array = tf.cast(image_array, tf.float32)
    return tf.keras.applications.densenet.preprocess_input(image_array)

def preprocess_resnet(image_array):
    image_array = tf.cast(image_array, tf.float32)
    return tf.keras.applications.resnet.preprocess_input(image_array)

# --- Plotting Functions --- #
def plot_training_history_st(history_dict, model_name):
    if history_dict is None:
        st.warning(f"No training history available for {model_name}.")
        return None

    hist_df = pd.DataFrame(history_dict)
    epochs = hist_df.index + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, hist_df['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in hist_df.columns:
        axes[0].plot(epochs, hist_df['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{model_name} Training and Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, hist_df['loss'], label='Training Loss')
    if 'val_loss' in hist_df.columns:
        axes[1].plot(epochs, hist_df['val_loss'], label='Validation Loss')
    axes[1].set_title(f'{model_name} Training and Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    return fig

def plot_confusion_matrix_st(cm, class_names, model_name):
    if cm is None:
        st.warning(f"No confusion matrix data available for {model_name}.")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Confusion Matrix for {model_name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    return fig

# --- Streamlit UI --- #
st.set_page_config(layout="wide")
st.title('Cat vs Dog Classifier & Model Analysis')

st.sidebar.header('Navigation')
page = st.sidebar.radio(
    'Go to',
    ['Predict Image', 'Model Comparison', 'Training History', 'Confusion Matrices']
)

if page == 'Predict Image':
    st.header('Predict an Image')
    st.write('Upload an image of a cat or a dog to get a prediction using different CNN models.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = np.asarray(image_resized)
        image_batch = np.expand_dims(image_array, axis=0)

        st.subheader('Select a Model for Prediction:')
        model_choice = st.radio(
            "",
            ('Simple CNN', 'DenseNet201', 'ResNet152'),
            index=0
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
                with st.spinner(f'Predicting with {model_name}...'):
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
            st.warning(f"The {model_choice} model could not be loaded. Please check the Google Drive ID.")
    else:
        st.info("Please upload an image to make a prediction.")

elif page == 'Model Comparison':
    st.header('Model Performance Comparison')
    if results_df is not None:
        st.dataframe(results_df.style.highlight_max(subset=['Test Accuracy'], axis=0).highlight_min(subset=['Test Loss'], axis=0))
        st.write("The table above shows the test loss and accuracy for each model.")
    else:
        st.warning("Model comparison results could not be loaded.")

elif page == 'Training History':
    st.header('Training History (Accuracy and Loss)')
    if training_histories is not None:
        for model_name, history_dict in training_histories.items():
            st.subheader(f'Training History for {model_name}')
            fig = plot_training_history_st(history_dict, model_name)
            if fig:
                st.pyplot(fig)
            else:
                st.warning(f"Could not plot history for {model_name}.")
    else:
        st.warning("Training histories could not be loaded.")

elif page == 'Confusion Matrices':
    st.header('Confusion Matrices')
    if confusion_matrices_data is not None:
        for model_name, data in confusion_matrices_data.items():
            st.subheader(f'Confusion Matrix for {model_name}')
            fig = plot_confusion_matrix_st(data['cm'], data['class_names'], model_name)
            if fig:
                st.pyplot(fig)
            else:
                st.warning(f"Could not plot confusion matrix for {model_name}.")
    else:
        st.warning("Confusion matrices data could not be loaded.")
