#!/usr/bin/env python3
"""
Enhanced Javanese Voice Recognition - Gradio Demo
Test the trained enhanced model through web interface
"""

import gradio as gr
import numpy as np
import tensorflow as tf
import librosa
import os
from pathlib import Path

# Model configuration
CLASSES = [
    'ha', 'na', 'ca', 'ra', 'ka', 'da', 'ta', 'sa', 'wa', 'la',
    'pa', 'dha', 'ja', 'ya', 'nya', 'ma', 'ga', 'ba', 'tha', 'nga'
]

# Load model globally
MODEL = None

def load_model():
    """Load the enhanced model"""
    global MODEL
    try:
        # Try new retrained model first
        model_path = 'models/javanese_enhanced_retrain.h5'
        if os.path.exists(model_path):
            MODEL = tf.keras.models.load_model(model_path)
            print("✅ Enhanced retrained model loaded successfully")
        else:
            # Fallback to old model
            MODEL = tf.keras.models.load_model('models/javanese_enhanced_model.h5')
            print("✅ Enhanced model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def extract_enhanced_features(y, sr=22050, target_len=2.0, n_mels=128):
    """Enhanced feature extraction matching training exactly"""
    # Normalize audio
    y = y.astype(np.float32)
    if len(y) == 0:
        y = np.zeros(int(target_len * sr))
    
    # Remove DC offset
    y = y - np.mean(y)
    
    # Apply pre-emphasis filter
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # Trim/pad to target length
    T = int(target_len * sr)
    if len(y) > T:
        # Take center segment for better stability
        start = (len(y) - T) // 2
        y = y[start:start + T]
    else:
        y = np.pad(y, (0, T - len(y)))
    
    # Apply window to reduce edge effects
    window = np.hanning(len(y))
    y = y * window
    
    # Compute mel-spectrogram with exact training parameters
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, 
        hop_length=512, win_length=2048,
        fmin=0, fmax=sr//2
    )
    
    # Convert to dB
    M = librosa.power_to_db(S, ref=np.max)
    
    # CMVN normalization
    M = (M - np.mean(M)) / (np.std(M) + 1e-8)
    
    return M.T.astype(np.float32)  # Transpose for time-first format

def predict_aksara(audio_file):
    """Predict Javanese aksara from audio file"""
    if MODEL is None:
        return "❌ Model not loaded!", {}
    
    if audio_file is None:
        return "❌ Please upload an audio file!", {}
    
    try:
        # Load audio with correct sample rate
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Extract features
        features = extract_enhanced_features(y, sr)
        
        # Prepare input for model
        X = features[None, :, :, None]  # Add batch dimension
        
        # Make prediction
        predictions = MODEL.predict(X, verbose=0)
        
        # Get results
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_idx]
        confidence = predictions[0][predicted_idx] * 100
        
        # Create confidence scores for all classes
        confidence_scores = {}
        for i, class_name in enumerate(CLASSES):
            confidence_scores[class_name] = float(predictions[0][i] * 100)
        
        # Sort by confidence
        sorted_scores = dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True))
        
        result = f"🎯 Predicted: **{predicted_class}** (Confidence: {confidence:.1f}%)"
        
        return result, sorted_scores
        
    except Exception as e:
        return f"❌ Error processing audio: {str(e)}", {}

def test_with_sample(class_name):
    """Test with sample from dataset"""
    if MODEL is None:
        return "❌ Model not loaded!", {}
    
    try:
        # Find a sample file
        sample_file = None
        for source in ['tts_generated', 'jv_manual_augmented']:
            source_path = Path('data') / source / class_name
            if source_path.exists():
                files = list(source_path.glob("*.wav"))
                if files:
                    sample_file = files[0]
                    break
        
        if sample_file is None:
            return f"❌ No sample found for {class_name}", {}
        
        # Load and predict
        y, sr = librosa.load(str(sample_file), sr=22050)
        features = extract_enhanced_features(y, sr)
        X = features[None, :, :, None]
        
        predictions = MODEL.predict(X, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_idx]
        confidence = predictions[0][predicted_idx] * 100
        
        # Create confidence scores
        confidence_scores = {}
        for i, class_name_iter in enumerate(CLASSES):
            confidence_scores[class_name_iter] = float(predictions[0][i] * 100)
        
        sorted_scores = dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True))
        
        status = "✅" if predicted_class == class_name else "❌"
        result = f"{status} Sample: {class_name} → Predicted: **{predicted_class}** (Confidence: {confidence:.1f}%)"
        
        return result, sorted_scores
        
    except Exception as e:
        return f"❌ Error: {str(e)}", {}

# Load model at startup
load_model()

# Create Gradio interface
with gr.Blocks(title="Enhanced Javanese Aksara Voice Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 Enhanced Javanese Aksara Voice Recognition
    
    **Model Performance**: 94.67% accuracy on test set | Enhanced CNN with improved regularization
    
    Upload an audio file (.wav, .mp3) or test with dataset samples to recognize Javanese aksara pronunciation.
    
    **Supported Aksara**: ha, na, ca, ra, ka, da, ta, sa, wa, la, pa, dha, ja, ya, nya, ma, ga, ba, tha, nga
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎤 Upload Audio")
            audio_input = gr.Audio(type="filepath", label="Audio File")
            predict_btn = gr.Button("🔮 Predict Aksara", variant="primary")
            
            gr.Markdown("### 📊 Test with Samples")
            sample_dropdown = gr.Dropdown(
                choices=CLASSES,
                label="Select Aksara",
                value="ha"
            )
            test_sample_btn = gr.Button("🧪 Test Sample", variant="secondary")
        
        with gr.Column():
            result_output = gr.Markdown("### 🎯 Prediction Result")
            confidence_plot = gr.Label(label="📊 Confidence Scores", num_top_classes=10)
    
    # Event handlers
    predict_btn.click(
        predict_aksara,
        inputs=[audio_input],
        outputs=[result_output, confidence_plot]
    )
    
    test_sample_btn.click(
        test_with_sample,
        inputs=[sample_dropdown],
        outputs=[result_output, confidence_plot]
    )
    
    gr.Markdown("""
    ### 📈 Model Information
    - **Architecture**: Enhanced CNN with GlobalAveragePooling2D + regularization
    - **Features**: 128-band mel-spectrogram (22050 Hz, 2 seconds)
    - **Parameters**: 210,164 trainable parameters
    - **Dataset**: Enhanced with data augmentation and class balancing
    - **Accuracy**: 94.67% test accuracy
    - **Model**: javanese_enhanced_retrain.h5
    """)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Javanese Voice Recognition Demo...")
    print("🌐 Open in browser: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)
