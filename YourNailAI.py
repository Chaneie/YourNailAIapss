import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time

# Configure page layout for full width
st.set_page_config(
    page_title="YourNailAI Detector",
    page_icon="🧬",
    layout="wide",  # This is crucial for full-width layout
    initial_sidebar_state="expanded"
)

# Load the model
model = load_model("mobilenetv2_bestmodel2.h5")

# Define classes and threshold for "Unknown" detection
class_names = ["Onychomycosis", "Psoriasis", "Unknown"]
class_descriptions = {
    "Onychomycosis": "Onychomycosis is a fungal infection of the nail that can cause discoloration and thickening.",
    "Psoriasis": "Nail Psoriasis is an inflammatory condition causing pitting, discoloration, or nail detachment.",
    "Unknown": "The image uploaded is not supported by this prototype."
}
threshold = 0.6

# Enhanced CSS styling with responsive design and full-width optimization
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Force full width usage */
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: none !important;
        width: 100% !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Responsive container */
    .app-container {
        width: 100%;
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        width: 100%;
    }
    
    .main-header h1 {
        color: white;
        font-weight: 700;
        font-size: clamp(2rem, 4vw, 3rem);
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: clamp(1rem, 2vw, 1.2rem);
        margin: 0;
    }
    
    /* Enhanced column layouts for different screen sizes */
    .upload-card, .results-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: fit-content;
        min-height: 500px;
        width: 100%;
    }
    
    .upload-card:hover, .results-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    .upload-zone {
        border: 3px dashed #667eea;
        border-radius: 16px;
        padding: clamp(20px, 4vw, 40px);
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        transition: all 0.3s ease;
        margin-bottom: 20px;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .upload-zone:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f4ff 0%, #e0ecff 100%);
        transform: scale(1.02);
    }
    
    .waiting-state {
        text-align: center;
        padding: clamp(30px, 6vw, 60px);
        color: #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        border-radius: 16px;
        border: 2px dashed rgba(102, 126, 234, 0.3);
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .waiting-state h3 {
        color: #667eea;
        margin-bottom: 15px;
        font-weight: 600;
        font-size: clamp(1.2rem, 3vw, 1.8rem);
    }
    
    .result-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
        text-align: center;
    }
    
    .result-unknown {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(255, 234, 167, 0.3);
        text-align: center;
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50px;
        padding: 4px;
        margin: 15px 0;
        max-width: 400px;
        width: 100%;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #00b894, #00cec9);
        height: 24px;
        border-radius: 50px;
        transition: width 1s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 14px;
        min-width: 60px;
    }
    
    .debug-panel, .metric-card, .expandable-section {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #6c5ce7;
        width: 100%;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
        margin: 5px 0;
        min-height: 50px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    .disease-icon {
        font-size: clamp(2rem, 5vw, 4rem);
        margin-bottom: 15px;
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Responsive image container - smaller fixed size */
    .image-container {
        width: 100%;
        max-width: 280px;
        margin: 0 auto 20px auto;
    }
    
    .image-container img {
        width: 250px !important;
        height: auto !important;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        display: block;
        margin: 0 auto;
    }
    
    /* Responsive grid for bottom section */
    .bottom-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 30px;
        margin-top: 40px;
        width: 100%;
    }
    
    /* Media queries for different screen sizes */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        
        .upload-card, .results-card {
            padding: 20px;
            min-height: 400px;
        }
        
        .main-header {
            padding: 30px 20px;
        }
        
        .bottom-grid {
            grid-template-columns: 1fr;
        }
    }
    
    @media (min-width: 1200px) {
        .main .block-container {
            max-width: 1400px !important;
            margin: 0 auto;
        }
    }
    
    /* Ensure sidebar doesn't interfere with main content width */
    .sidebar .stSelectbox, .sidebar .stSlider, .sidebar .stCheckbox {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.markdown(
    """
    <div class="main-header">
        <h1>🧬 YourNailAI Detector</h1>
        <p>Advanced AI-powered nail disease detection using MobileNetV2</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Debug options in sidebar with modern styling
st.sidebar.markdown("### ⚙ Configuration")
show_debug = st.sidebar.checkbox("🔍 Debug Mode", value=False)
custom_threshold = st.sidebar.slider("🎯 Confidence Threshold", 0.1, 0.9, 0.6, 0.1)
preprocessing_method = st.sidebar.selectbox("🔧 Preprocessing Method", 
                                          ["Standard (0-1)", "ImageNet Normalization", "Custom"])

# Initialize session state
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = None
if 'analyzing' not in st.session_state:
    st.session_state.analyzing = False

# Create responsive columns with proper spacing
# Use different column ratios based on content
col1, col2 = st.columns([0.9, 1.1], gap="large")

# Left column - Upload section
with col1:
    st.markdown("### 📸 Upload Image")
    
    # Custom file uploader with better styling
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "png", "jpeg"],
        help="Upload a clear image of the nail",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(uploaded_file, caption="📋 Uploaded Image", width=350)  # Fixed smaller size
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Button container for better spacing
        st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
        
        # Create button columns for better layout
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            detect_button = st.button("🚀 Analyze Now")
        with btn_col2:
            clear_button = st.button("🗑 Clear Image")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if clear_button:
            st.session_state.detection_results = None
            st.session_state.debug_info = None
            st.session_state.analyzing = False
            st.experimental_rerun()
    else:
        # Show upload zone when no file is uploaded
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem; margin-bottom: 15px;">📤</div>
            <h3>Drag and Drop</h3>
            <p>or click to upload an image</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def preprocess_image(image, method="Standard (0-1)"):
    """Preprocess image with different methods"""
    image = image.resize((250, 250))
    image_array = np.array(image)
    
    if method == "Standard (0-1)":
        image_array = image_array / 255.0
    elif method == "ImageNet Normalization":
        image_array = image_array / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
    elif method == "Custom":
        image_array = (image_array - 127.5) / 127.5
    
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Right column - Results section
with col2:
    st.markdown("### 🎯 Analysis Results")
    
    # Handle detection
    if uploaded_file and 'detect_button' in locals() and detect_button:
        st.session_state.analyzing = True
        
        # Show loading animation
        with st.spinner("🔬 Analyzing image with AI..."):
            time.sleep(2)
            
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = preprocess_image(image, preprocessing_method)
            predictions = model.predict(image_array)
            raw_predictions = predictions[0]
            
            # Store debug info
            st.session_state.debug_info = {
                'image_shape': image_array.shape,
                'raw_predictions': raw_predictions,
                'prediction_shape': predictions.shape,
                'sum_predictions': np.sum(raw_predictions),
                'class_confidences': [(class_names[i], raw_predictions[i] if len(raw_predictions) > i else 0) 
                                    for i in range(len(class_names))]
            }
            
            # Interpret predictions for 3-class classification
            # Model outputs 2 classes but we want to classify into 3 (including Unknown)
            if len(raw_predictions) == 1:
                # Binary classification output (sigmoid)
                confidence_class_1 = raw_predictions[0]  # Psoriasis
                confidence_class_0 = 1 - confidence_class_1  # Onychomycosis
                
                # Determine predicted class and confidence
                if confidence_class_1 > 0.5:
                    predicted_class = 1  # Psoriasis
                    max_confidence = confidence_class_1
                else:
                    predicted_class = 0  # Onychomycosis
                    max_confidence = confidence_class_0
            else:
                # Multi-class output (softmax)
                predicted_class = np.argmax(raw_predictions)
                max_confidence = np.max(raw_predictions)
            
            # Apply threshold for "Unknown" classification
            # If confidence is below threshold, classify as "Unknown"
            if max_confidence < custom_threshold:
                final_class = "Unknown"
                final_confidence = max_confidence
            else:
                final_class = class_names[predicted_class]
                final_confidence = max_confidence

            st.session_state.detection_results = {
                'final_class': final_class,
                'final_confidence': final_confidence,
                'predicted_class': predicted_class,
                'max_confidence': max_confidence,
                'raw_predictions': raw_predictions
            }
            st.session_state.analyzing = False
    
    # Display results
    if st.session_state.detection_results:
        results = st.session_state.detection_results
        
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        
        # Display result based on final classification
        if results['final_class'] == "Unknown":
            st.markdown(f"""
            <div class="result-unknown">
                <div class="disease-icon">❓</div>
                <h2 style="margin: 0; font-size: clamp(1.3rem, 3vw, 2rem);">Unknown Condition</h2>
                <p style="margin: 10px 0; font-size: clamp(1rem, 2vw, 1.2rem);">
                    Confidence: {results['final_confidence']*100:.1f}% (Below {custom_threshold*100:.0f}% threshold)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence bar for unknown
            confidence_percentage = results['final_confidence'] * 100
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_percentage}%; background: linear-gradient(90deg, #fab1a0, #ffeaa7);">
                        {confidence_percentage:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"💡 Description:** {class_descriptions['Unknown']}")
            st.info("💡 This image may not show a detectable nail condition or may be outside our model's training scope. Try uploading a clearer image or consult a medical professional.")
            
        else:
            # Known disease detected
            disease_name = results['final_class']
            disease_emoji = "🦠" if disease_name == "Onychomycosis" else "🔴"
            
            st.markdown(f"""
            <div class="result-success">
                <div class="disease-icon">{disease_emoji}</div>
                <h2 style="margin: 0; font-size: clamp(1.3rem, 3vw, 2rem);">{disease_name}</h2>
                <p style="margin: 10px 0; opacity: 0.9; font-size: clamp(1rem, 2vw, 1.2rem);">
                    Confidence: {results['final_confidence']*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Animated confidence bar with centering
            confidence_percentage = results['final_confidence'] * 100
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_percentage}%;">
                        {confidence_percentage:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"💡 Description:** {class_descriptions[disease_name]}")
            
            if results['final_confidence'] < 0.8:
                st.warning("⚠ *Moderate confidence* - Please consult a medical professional for verification.")
            else:
                st.success("✅ *High confidence* - Consider medical consultation for treatment options.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Debug information with modern styling
        if show_debug and st.session_state.debug_info:
            debug = st.session_state.debug_info
            st.markdown('<div class="debug-panel">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Debug Information")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(f"*Image Shape:* {debug['image_shape']}")
                st.markdown(f"*Prediction Shape:* {debug['prediction_shape']}")
            with col_d2:
                st.markdown(f"*Sum of Predictions:* {debug['sum_predictions']:.4f}")
                st.markdown(f"*Raw Predictions:* {debug['raw_predictions']}")
            
            st.markdown("*Individual Class Confidences:*")
            for class_name, confidence in debug['class_confidences']:
                st.markdown(f"- *{class_name}:* {confidence:.4f}")
            
            st.markdown(f"*Final Classification:* {results['final_class']} (Confidence: {results['final_confidence']:.4f})")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif uploaded_file and not st.session_state.analyzing:
        st.markdown("""
        <div class="waiting-state pulse">
            <div style="font-size: 3rem; margin-bottom: 15px;">🎯</div>
            <h3>Ready to Analyze</h3>
            <p>Click <strong>'Analyze Now'</strong> to start AI detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.analyzing:
        st.markdown("""
        <div class="waiting-state">
            <div style="font-size: 3rem; margin-bottom: 15px;">🔬</div>
            <h3>Analyzing...</h3>
            <p>Our AI is examining your image</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="waiting-state">
            <div style="font-size: 3rem; margin-bottom: 15px;">📤</div>
            <h3>Upload an Image</h3>
            <p>Upload a nail image to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Bottom section with model info using responsive grid
st.markdown("---")

# Use HTML grid for better responsive layout
st.markdown('<div class="bottom-grid">', unsafe_allow_html=True)

# Final disclaimer
st.markdown(
    """
    <div style="text-align: center; margin-top: 40px; padding: 20px;">
        <p style="color: rgba(255, 255, 255, 0.8); font-size: clamp(0.9rem, 2vw, 1.1rem);">
            ⚠ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)