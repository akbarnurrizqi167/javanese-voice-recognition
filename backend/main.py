from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, io, librosa
from pydub import AudioSegment

# 20 Aksara Jawa Lengkap (Alphabetical order matching retrained model)
CLASSES = ["ba", "ca", "da", "dha", "ga", "ha", "ja", "ka", "la", "ma", "na", "nga", "nya", "pa", "ra", "sa", "ta", "tha", "wa", "ya"]

def to_wav_mono_bytes(raw_bytes: bytes) -> np.ndarray:
    # Convert various container formats (webm/ogg/mp3/mp4) to mono float32 waveform.
    # Use 22050 Hz to match training data
    seg = AudioSegment.from_file(io.BytesIO(raw_bytes))
    seg = seg.set_frame_rate(22050).set_channels(1).set_sample_width(2)  # 16-bit PCM
    samples = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples

def extract_enhanced_features(y, sr=22050, target_len=2.0, n_mels=128):
    """Enhanced feature extraction matching the retrained model exactly"""
    # Normalize audio
    y = y.astype(np.float32)
    if len(y) == 0:
        y = np.zeros(int(target_len * sr))
    
    # Remove DC offset
    y = y - np.mean(y)
    
    # Trim/pad to target length (2 seconds to match training)
    T = int(target_len * sr)  # 44100 samples for 2 seconds
    if len(y) > T:
        # Take center segment for better stability
        start = (len(y) - T) // 2
        y = y[start:start + T]
    else:
        y = np.pad(y, (0, T - len(y)))
    
    # Apply pre-emphasis filter (matching training)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # Apply window to reduce edge effects
    window = np.hanning(len(y))
    y = y * window
    
    # Extract mel-spectrogram (matching training parameters)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, 
        hop_length=512, win_length=2048,
        fmin=0, fmax=sr//2
    )
    
    # Convert to dB and normalize (matching training)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # CMVN (Cepstral Mean and Variance Normalization)
    mel_spec_normalized = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
    
    # Return in format expected by model: [batch, time, freq, channels]
    return mel_spec_normalized.T[None, :, :, None].astype(np.float32)

def infer_probs_demo(feat):
    # DEMO MODE: random logits for 20 classes
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(len(CLASSES),)).astype(np.float32)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs

# Load enhanced trained model
try:
    import tensorflow as tf
    # Prioritize enhanced retrained model paths
    model_paths = [
        '../models/javanese_enhanced_retrain.h5',  # New retrained model (94.67% accuracy)
        'models/javanese_enhanced_retrain.h5',
        '../models/javanese_enhanced_model.h5',    # Old enhanced model (overfitted)
        'models/javanese_enhanced_model.h5',
        '../models/javanese_20_model.h5',          # Fallback to basic model
        '../models/best_javanese_model.h5', 
        'models/javanese_20_model.h5',
        '../best_javanese_model_20.h5',
        '../best_javanese_model.h5'
    ]
    
    model = None
    model_type = "unknown"
    for path in model_paths:
        try:
            model = tf.keras.models.load_model(path)
            if "retrain" in path:
                model_type = "retrained"
                print(f"✅ Retrained model loaded from: {path} (94.67% accuracy)")
            elif "enhanced" in path:
                model_type = "enhanced"
                print(f"✅ Enhanced model loaded from: {path} (92% accuracy)")
            else:
                model_type = "basic"
                print(f"✅ Basic model loaded from: {path}")
            break
        except Exception as e:
            continue
    
    MODEL_LOADED = model is not None
    if not MODEL_LOADED:
        print("⚠️ No trained model found. Using demo mode for API.")
        print("📁 Place trained models in models/ directory")
        print("🎯 For best performance, use: javanese_enhanced_model.h5")
        model_type = "demo"
        
except Exception as e:
    print(f"⚠️ TensorFlow error: {e}. Using demo mode for API.")
    MODEL_LOADED = False
    model = None
    model_type = "demo"

def infer_probs(feat):
    """Enhanced inference function using trained model if available"""
    if MODEL_LOADED:
        predictions = model.predict(feat, verbose=0)
        return predictions[0]
    else:
        return infer_probs_demo(feat)

app = FastAPI(
    title="Enhanced Javanese Aksara Recognition API", 
    version="2.2",
    description="""
    🎯 **Javanese Voice Recognition API**
    
    Advanced AI-powered API for recognizing Javanese aksara from audio input.
    
    ## Features
    - **20 Aksara Support**: Complete Javanese aksara recognition
    - **Enhanced CNN Model**: 94.67% accuracy with retrained architecture
    - **Real-time Processing**: Fast audio-to-text conversion
    - **Batch Processing**: Multiple files at once
    - **RESTful Design**: Standard HTTP methods and JSON responses
    
    ## Supported Aksara
    ba, ca, da, dha, ga, ha, ja, ka, la, ma, na, nga, nya, pa, ra, sa, ta, tha, wa, ya
    
    ## Usage
    1. Upload audio file (WAV, MP3, WebM, OGG)
    2. Specify target aksara for validation
    3. Get prediction with confidence scores
    """,
    contact={
        "name": "Javanese Voice Demo",
        "url": "http://localhost:8001"
    }
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/", tags=["Info"])
def root():
    """API Root - Welcome message and basic info"""
    return {
        "message": "🎯 Javanese Aksara Voice Recognition API",
        "version": "2.2_Enhanced", 
        "model": f"{model_type} (94.67% accuracy)" if MODEL_LOADED else "demo",
        "endpoints": {
            "health": "/health",
            "classes": "/classes", 
            "predict": "/predict?target=<aksara>",
            "batch": "/batch_predict",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "supported_formats": ["WAV", "MP3", "WebM", "OGG"],
        "aksara_count": len(CLASSES)
    }

@app.get("/health", tags=["Monitoring"])
def health():
    """Health check endpoint - Monitor API and model status"""
    return {
        "status": "ok", 
        "classes": len(CLASSES),
        "model_loaded": MODEL_LOADED,
        "model_type": model_type,
        "aksara": CLASSES,
        "version": "2.2_Enhanced",
        "description": "20 Javanese Aksara with Enhanced CNN",
        "accuracy": "94.67%" if model_type == "retrained" else "92%" if model_type == "enhanced" else "87%" if model_type == "basic" else "demo",
        "sample_rate": "22050 Hz",
        "duration": "2 seconds",
        "features": "Mel-spectrogram + CMVN"
    }

@app.get("/classes", tags=["Info"])
def get_classes():
    """Get available aksara classes with detailed info"""
    return {
        "classes": CLASSES,
        "count": len(CLASSES),
        "description": "20 Javanese aksara for voice recognition",
        "alphabet_order": True,
        "categories": {
            "consonants": ["ba", "ca", "da", "dha", "ga", "ha", "ja", "ka", "la", "ma", "na", "nga", "nya", "pa", "ra", "sa", "ta", "tha", "wa", "ya"],
            "total": len(CLASSES)
        }
    }

@app.post("/predict", tags=["Recognition"])
async def predict(target: str, file: UploadFile = File(...)):
    """
    Predict Javanese aksara from audio file
    
    - **target**: Target aksara to validate against (required)
    - **file**: Audio file (WAV, MP3, WebM, OGG)
    
    Returns prediction with confidence scores and validation status.
    """
    raw = await file.read()
    # Ubah ke waveform 22050 mono (matching training)
    try:
        y = to_wav_mono_bytes(raw)
    except Exception as e:
        return {"error": f"Audio tidak didukung: {type(e).__name__}: {e}"}
    
    feat = extract_enhanced_features(y, sr=22050)
    probs = infer_probs(feat)  # Use enhanced inference function
    pred_idx = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]
    conf = float(probs[pred_idx])

    score, status = None, None
    if target in CLASSES:
        t_idx = CLASSES.index(target)
        target_conf = float(probs[t_idx])
        score = round(100*target_conf, 1)
        status = "✅ Benar" if score >= 70 else ("⚠️ Hampir" if score >= 50 else "❌ Coba lagi")
    else:
        # Target not in our 20 classes
        return {"error": f"Target '{target}' not in supported classes: {CLASSES}"}

    top5_idx = np.argsort(-probs)[:5]
    top5 = [{"label": CLASSES[i], "p": float(probs[i])} for i in top5_idx]

    return {
        "pred": pred_label, 
        "conf": round(conf, 3), 
        "score": score, 
        "status": status, 
        "top5": top5,
        "model_mode": f"Enhanced AI ({model_type})" if MODEL_LOADED else "Demo",
        "model_accuracy": "94.67%" if model_type == "retrained" else "92%" if model_type == "enhanced" else "87%" if model_type == "basic" else "demo",
        "classes_used": "20_javanese_aksara_enhanced",
        "file_info": {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(raw)
        }
    }

@app.post("/batch_predict", tags=["Recognition"])
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Batch prediction for multiple audio files
    
    Upload multiple audio files for bulk processing.
    Returns predictions for each file with success/error status.
    """
    results = []
    
    for file in files:
        raw = await file.read()
        try:
            y = to_wav_mono_bytes(raw)
            feat = extract_enhanced_features(y, sr=22050)
            probs = infer_probs(feat)
            
            pred_idx = int(np.argmax(probs))
            pred_label = CLASSES[pred_idx]
            conf = float(probs[pred_idx])
            
            # Get top 3 for batch processing
            top3_idx = np.argsort(-probs)[:3]
            top3 = [{"label": CLASSES[i], "p": float(probs[i])} for i in top3_idx]
            
            results.append({
                "filename": file.filename,
                "pred": pred_label,
                "conf": round(conf, 3),
                "top3": top3,
                "size_bytes": len(raw),
                "success": True
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {
        "results": results, 
        "total": len(files),
        "successful": len([r for r in results if r.get("success", False)]),
        "failed": len([r for r in results if not r.get("success", True)]),
        "model_info": {
            "type": model_type,
            "accuracy": "94.67%" if model_type == "retrained" else "92%" if model_type == "enhanced" else "87%"
        }
    }

@app.post("/predict_simple", tags=["Recognition"])
async def predict_simple(file: UploadFile = File(...)):
    """
    Simple prediction without target validation
    
    Just predict the aksara without comparing to target.
    Useful for general recognition tasks.
    """
    raw = await file.read()
    try:
        y = to_wav_mono_bytes(raw)
        feat = extract_enhanced_features(y, sr=22050)
        probs = infer_probs(feat)
        
        pred_idx = int(np.argmax(probs))
        pred_label = CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        
        top3_idx = np.argsort(-probs)[:3]
        top3 = [{"label": CLASSES[i], "p": round(float(probs[i]), 4)} for i in top3_idx]
        
        return {
            "prediction": pred_label,
            "confidence": round(conf, 4),
            "top3": top3,
            "model": f"{model_type} (94.67%)" if MODEL_LOADED else "demo"
        }
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

@app.get("/model/info", tags=["Monitoring"])
def model_info():
    """Get detailed model information"""
    if MODEL_LOADED:
        return {
            "model_loaded": True,
            "model_type": model_type,
            "accuracy": "94.67%" if model_type == "retrained" else "92%" if model_type == "enhanced" else "87%",
            "architecture": "Enhanced CNN",
            "input_shape": "(None, 87, 128, 1)",
            "output_classes": len(CLASSES),
            "sample_rate": "22050 Hz",
            "duration": "2.0 seconds",
            "features": "Mel-spectrogram + CMVN normalization",
            "regularization": "L2 + Dropout + BatchNorm",
            "training_data": "Javanese manual + augmented dataset"
        }
    else:
        return {
            "model_loaded": False,
            "mode": "demo",
            "message": "No trained model available"
        }

@app.get("/stats", tags=["Monitoring"])
def get_stats():
    """Get API usage statistics"""
    return {
        "api_version": "2.2_Enhanced",
        "supported_classes": len(CLASSES),
        "model_status": "active" if MODEL_LOADED else "demo",
        "features": {
            "real_time_prediction": True,
            "batch_processing": True,
            "multiple_formats": True,
            "confidence_scoring": True,
            "top_k_results": True
        },
        "technical_specs": {
            "sample_rate": "22050 Hz",
            "duration": "2 seconds", 
            "feature_extraction": "Mel-spectrogram",
            "normalization": "CMVN",
            "model_architecture": "CNN + GlobalAveragePooling"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
