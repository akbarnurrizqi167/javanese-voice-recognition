"""
Javanese Aksara Voice Recognition API - Railway Production
FastAPI backend with real ML model for Javanese voice recognition
"""

import os
import io
import logging
from datetime import datetime
from typing import Optional, List
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import numpy as np
    import librosa
    import tensorflow as tf
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.middleware.cors import CORSMiddleware
    from pydub import AudioSegment
    import pickle
    logger.info("✅ All ML dependencies loaded successfully")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    raise e

# Initialize FastAPI
app = FastAPI(
    title="Javanese Aksara Voice Recognition API",
    description="Real ML-powered API for recognizing Javanese aksara from voice",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
label_encoder = None
start_time = datetime.now()
prediction_count = 0

# Javanese aksara classes
CLASSES = [
    'ha', 'na', 'ca', 'ra', 'ka', 'da', 'ta', 'sa', 'wa', 'la',
    'pa', 'dha', 'ja', 'ya', 'nya', 'ma', 'ga', 'ba', 'tha', 'nga'
]

def load_model_and_encoder():
    """Load TensorFlow model and label encoder"""
    global model, label_encoder
    
    try:
        # Load model
        model_path = "models/javanese_enhanced_retrain.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"✅ Enhanced model loaded: {model.count_params()} parameters")
        else:
            logger.error(f"❌ Model file not found: {model_path}")
            return False
        
        # Load label encoder
        encoder_path = "models/label_encoder_retrain.pkl"
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            logger.info("✅ Label encoder loaded")
        else:
            logger.warning("⚠️ Label encoder not found, using CLASSES array")
            
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model/encoder: {e}")
        return False

def extract_enhanced_features(y, sr=22050, target_len=2.0, n_mels=128):
    """Enhanced feature extraction matching training exactly"""
    try:
        # Normalize audio
        y = y.astype(np.float32)
        if len(y) == 0:
            y = np.zeros(int(target_len * sr))
        
        # Remove DC offset
        y = y - np.mean(y)
        
        # Apply pre-emphasis filter
        if len(y) > 1:
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
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

def convert_audio_to_wav(audio_file):
    """Convert audio file to WAV format"""
    try:
        audio_bytes = audio_file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(22050)
        
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        return wav_buffer
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("🚀 Starting Javanese Aksara Recognition API with Real ML Model...")
    try:
        success = load_model_and_encoder()
        if success:
            logger.info("✅ API ready for real ML predictions")
        else:
            logger.warning("⚠️ API started but model not loaded")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🎯 Javanese Aksara Voice Recognition API",
        "version": "2.0.0",
        "model": "Enhanced CNN (94.67% accuracy)",
        "status": "production",
        "docs": "/docs",
        "model_loaded": model is not None,
        "supported_aksara": len(CLASSES),
        "platform": "Railway"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global start_time, prediction_count
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "predictions_made": prediction_count,
        "model_loaded": model is not None,
        "model_type": "Enhanced CNN" if model else "None",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        return {
            "status": "Model not loaded",
            "supported_aksara": CLASSES,
            "classes": len(CLASSES)
        }
    
    return {
        "model_name": "Enhanced Javanese Aksara CNN",
        "version": "2.0.0",
        "accuracy": 94.67,
        "parameters": model.count_params(),
        "classes": len(CLASSES),
        "supported_aksara": CLASSES,
        "input_format": {
            "sample_rate": 22050,
            "duration": "2.0 seconds",
            "features": "128-band mel-spectrogram",
            "preprocessing": "CMVN + pre-emphasis"
        },
        "architecture": "CNN with GlobalAveragePooling2D"
    }

@app.get("/supported-aksara")
async def supported_aksara():
    """Get list of supported Javanese aksara"""
    return {
        "count": len(CLASSES),
        "aksara": CLASSES,
        "description": "20 traditional Javanese aksara characters",
        "note": "Real ML predictions with 94.67% accuracy"
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    target: Optional[str] = Form(None)
):
    """Predict Javanese aksara from audio file using real ML model"""
    global prediction_count
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded - check server logs")
    
    try:
        # Convert audio
        wav_buffer = convert_audio_to_wav(file.file)
        
        # Load audio with librosa
        audio_data, sr = librosa.load(wav_buffer, sr=22050)
        
        # Extract features using enhanced method
        features = extract_enhanced_features(audio_data, sr)
        
        # Prepare input for model
        X = features[None, :, :, None]  # Add batch and channel dimensions
        
        # Make prediction
        predictions = model.predict(X, verbose=0)
        
        # Get results
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_idx]
        confidence = float(predictions[0][predicted_idx])
        
        # Create top predictions
        top_predictions = []
        for i, class_name in enumerate(CLASSES):
            top_predictions.append({
                "class": class_name,
                "confidence": float(predictions[0][i])
            })
        
        top_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        prediction_count += 1
        
        result = {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "target": target,
            "is_correct": predicted_class == target if target else None,
            "top_predictions": top_predictions[:5],
            "model_info": {
                "model": "Enhanced CNN",
                "accuracy": "94.67%",
                "features": "128-band mel-spectrogram"
            },
            "metadata": {
                "audio_duration": len(audio_data) / sr,
                "sample_rate": sr,
                "feature_shape": features.shape,
                "model_version": "2.0.0",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Real prediction: {predicted_class} (conf: {confidence:.3f}, target: {target})")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Test file upload and audio processing without prediction"""
    try:
        # Convert and analyze audio
        wav_buffer = convert_audio_to_wav(file.file)
        audio_data, sr = librosa.load(wav_buffer, sr=22050)
        
        # Extract features to test preprocessing
        features = extract_enhanced_features(audio_data, sr)
        
        return {
            "success": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(await file.read()),
            "audio_info": {
                "duration": len(audio_data) / sr,
                "sample_rate": sr,
                "samples": len(audio_data),
                "feature_shape": features.shape
            },
            "status": "upload_and_preprocessing_successful"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload test failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    global start_time, prediction_count
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "total_predictions": prediction_count,
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m",
        "start_time": start_time.isoformat(),
        "model_status": "loaded" if model else "not_loaded",
        "model_type": "Enhanced CNN",
        "accuracy": "94.67%",
        "supported_classes": len(CLASSES),
        "platform": "Railway",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting production server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
