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
import base64

# --- Page Config with Custom Theme ---
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Colorful 3D-like Styling ---
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Colorful Animated Title */
    .rainbow-text {
        background: linear-gradient(90deg, #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #9400d3);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rainbow 3s linear infinite;
        font-weight: 700;
        font-size: 2.5em;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    @keyframes rainbow {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    /* 3D-like Buttons */
    .stButton>button, div[data-testid="stSelectbox"] > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4), 
                    0 6px 6px rgba(102, 126, 234, 0.3),
                    inset 0 -2px 0 rgba(0,0,0,0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover, div[data-testid="stSelectbox"] > div:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.5),
                    0 15px 12px rgba(102, 126, 234, 0.4),
                    inset 0 -2px 0 rgba(0,0,0,0.2) !important;
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%) !important;
    }
    
    .stButton>button:active {
        transform: translateY(1px) scale(0.98) !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    }
    
    .css-1d391kg .rainbow-text {
        font-size: 1.5em;
    }
    
    /* Selectbox Dropdown Styling */
    div[data-testid="stSelectbox"] label {
        color: #fff !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Image Upload Animation */
    @keyframes floatUpload {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .upload-animation {
        animation: floatUpload 3s ease-in-out infinite;
    }
    
    /* Success Message Styling */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        border-radius: 15px !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.4) !important;
    }
    
    /* Prediction Result Box */
    .prediction-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(250, 112, 154, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Spinner Styling */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent !important;
    }
    
    /* Table Styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        overflow: hidden !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Radio Buttons in Sidebar */
    .stRadio > label {
        color: #fff !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 5px !important;
    }
    
    /* File Uploader Styling */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 2px dashed rgba(255, 255, 255, 0.6) !important;
        border-radius: 20px !important;
        color: white !important;
    }
    
    .stFileUploader > div > div:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        border-color: #fff !important;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

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
        st.error(f"Google Drive File ID for {file_key} is not configured.")
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
    fig.patch.set_facecolor('none')
    axes[0].set_facecolor('none')
    axes[1].set_facecolor('none')

    axes[0].plot(epochs, hist_df['accuracy'], label='Training Accuracy', color='#667eea', linewidth=2)
    if 'val_accuracy' in hist_df.columns:
        axes[0].plot(epochs, hist_df['val_accuracy'], label='Validation Accuracy', color='#f093fb', linewidth=2)
    axes[0].set_title(f'{model_name} Accuracy', color='white', fontweight='bold')
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].set_ylabel('Accuracy', color='white')
    axes[0].legend(loc='best', framealpha=0.3, edgecolor='white', labelcolor='white')
    axes[0].tick_params(colors='white')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, hist_df['loss'], label='Training Loss', color='#11998e', linewidth=2)
    if 'val_loss' in hist_df.columns:
        axes[1].plot(epochs, hist_df['val_loss'], label='Validation Loss', color='#38ef7d', linewidth=2)
    axes[1].set_title(f'{model_name} Loss', color='white', fontweight='bold')
    axes[1].set_xlabel('Epoch', color='white')
    axes[1].set_ylabel('Loss', color='white')
    axes[1].legend(loc='best', framealpha=0.3, edgecolor='white', labelcolor='white')
    axes[1].tick_params(colors='white')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_confusion_matrix_st(cm, class_names, model_name):
    if cm is None:
        st.warning(f"No confusion matrix data available for {model_name}.")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={'color': 'white', 'fontsize': 12})
    ax.set_title(f'Confusion Matrix - {model_name}', color='white', fontweight='bold', fontsize=14)
    ax.set_xlabel('Predicted Label', color='white', fontsize=12)
    ax.set_ylabel('True Label', color='white', fontsize=12)
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    return fig

# --- Animated Header Component ---
def animated_header():
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 class="rainbow-text">🐱 Cat vs Dog Classifier 🐶</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.2em; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
            Deep Learning Model Comparison with CNN, DenseNet201 & ResNet152
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- Streamlit UI --- #
animated_header()

st.sidebar.markdown("""
    <h2 style="color: white; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
        🧭 Navigation
    </h2>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    '',
    ['Predict Image', 'Model Comparison', 'Training History', 'Confusion Matrices'],
    label_visibility='collapsed'
)

if page == 'Predict Image':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader('📤 Upload an Image')
    st.write('Upload an image of a cat or a dog to get a prediction.')
    
    uploaded_file = st.file_uploader("Drag and drop or click to upload...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="upload-animation">', unsafe_allow_html=True)
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
            image_array = np.asarray(image_resized)
            image_batch = np.expand_dims(image_array, axis=0)
            
            st.subheader('🎯 Select Model')
            model_choice = st.selectbox(
                "Choose a model for prediction",
                ('Simple CNN', 'DenseNet201', 'ResNet152'),
                index=0
            )
            
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
                with st.spinner(f'🧠 Analyzing with {model_name}...'):
                    processed_image_batch = preprocess_func(image_batch)
                    prediction = model_to_use.predict(processed_image_batch)
                    predicted_class_index = (prediction > 0.5).astype(int)[0][0]
                    predicted_class = class_names[predicted_class_index]
                    confidence = prediction[0][0] if predicted_class_index == 1 else (1 - prediction[0][0])
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: white; margin: 0;">🎉 Result</h2>
                    <h1 style="color: white; margin: 10px 0; font-size: 2.5em;">
                        {predicted_class.upper()}
                    </h1>
                    <p style="color: rgba(255,255,255,0.9); font-size: 1.3em; margin: 0;">
                        Confidence: <b>{confidence:.2%}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ The {model_choice} model could not be loaded.")
    else:
        st.info("👆 Please upload an image to make a prediction.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == 'Model Comparison':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader('📊 Model Performance Comparison')
    
    if results_df is not None:
        st.dataframe(
            results_df.style
            .highlight_max(subset=['Test Accuracy'], color='#38ef7d')
            .highlight_min(subset=['Test Loss'], color='#fa709a')
            .set_properties(**{'background-color': 'rgba(255,255,255,0.1)', 'color': 'white'})
        )
        st.write("The table above shows the test loss and accuracy for each model.")
    else:
        st.warning("Model comparison results could not be loaded.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == 'Training History':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader('📈 Training History')
    
    if training_histories is not None:
        for model_name, history_dict in training_histories.items():
            st.markdown(f'<h4 style="color: white;">{model_name}</h4>', unsafe_allow_html=True)
            fig = plot_training_history_st(history_dict, model_name)
            if fig:
                st.pyplot(fig, transparent=True)
            else:
                st.warning(f"Could not plot history for {model_name}.")
    else:
        st.warning("Training histories could not be loaded.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == 'Confusion Matrices':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader('🎯 Confusion Matrices')
    
    if confusion_matrices_data is not None:
        for model_name, data in confusion_matrices_data.items():
            st.markdown(f'<h4 style="color: white;">{model_name}</h4>', unsafe_allow_html=True)
            fig = plot_confusion_matrix_st(data['cm'], data['class_names'], model_name)
            if fig:
                st.pyplot(fig, transparent=True)
            else:
                st.warning(f"Could not plot confusion matrix for {model_name}.")
    else:
        st.warning("Confusion matrices data could not be loaded.")
    
    st.markdown('</div>', unsafe_allow_html=True)
