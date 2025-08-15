from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, io, librosa
from pydub import AudioSegment

# 20 Aksara Jawa Lengkap
CLASSES = ["ha", "na", "ca", "ra", "ka", "da", "ta", "sa", "wa", "la", "pa", "dha", "ja", "ya", "nya", "ma", "ga", "ba", "tha", "nga"]

def to_wav16k_mono_bytes(raw_bytes: bytes) -> np.ndarray:
    # Convert various container formats (webm/ogg/mp3/mp4) to mono 16k float32 waveform.
    seg = AudioSegment.from_file(io.BytesIO(raw_bytes))
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16-bit PCM
    samples = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples

def extract_features(y, sr=16000, target_len=1.0, n_mels=64):
    T = int(target_len*sr)
    if len(y) > T: y = y[:T]
    else: y = np.pad(y, (0, T-len(y)))
    
    # === Fixed: Manual mel-spectrogram untuk kompatibilitas librosa ===
    n_fft = 400
    hop = 160
    # STFT -> power spec
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop, center=False))**2
    # Mel filterbank
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    m = np.dot(mel_filter, S)
    
    # log + CMVN
    m = np.log(m + 1e-6)
    m = (m - m.mean()) / (m.std() + 1e-9)
    return m.astype(np.float32)[None, :, :, None]

def infer_probs_demo(feat):
    # DEMO MODE: random logits for 20 classes
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(len(CLASSES),)).astype(np.float32)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs

# Load trained model (jika ada)
try:
    import tensorflow as tf
    # Try different model paths
    model_paths = [
        '../models/javanese_20_model.h5',
        '../best_javanese_model_20.h5',
        'models/javanese_20_model.h5',
        '../best_javanese_model.h5'
    ]
    
    model = None
    for path in model_paths:
        try:
            model = tf.keras.models.load_model(path)
            print(f"✅ Trained model loaded from: {path}")
            break
        except:
            continue
    
    MODEL_LOADED = model is not None
    if not MODEL_LOADED:
        print("⚠️ No trained model found. Using demo mode for API.")
        
except:
    print("⚠️ TensorFlow not available. Using demo mode for API.")
    MODEL_LOADED = False
    model = None

def infer_probs(feat):
    """Inference function that uses trained model if available, otherwise demo mode"""
    if MODEL_LOADED:
        predictions = model.predict(feat, verbose=0)
        return predictions[0]
    else:
        return infer_probs_demo(feat)

app = FastAPI(title="Javanese Aksara TTS API", version="2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "classes": len(CLASSES),
        "model_loaded": MODEL_LOADED,
        "aksara": CLASSES,
        "version": "2.1_TTS",
        "description": "20 Javanese Aksara"
    }

@app.get("/classes")
def get_classes():
    """Get available aksara classes"""
    return {
        "classes": CLASSES,
        "count": len(CLASSES),
        "description": "20 Javanese aksara for TTS training"
    }

@app.post("/predict")
async def predict(target: str, file: UploadFile = File(...)):
    raw = await file.read()
    # Ubah ke waveform 16k mono
    try:
        y = to_wav16k_mono_bytes(raw)
    except Exception as e:
        return {"error": f"Audio tidak didukung: {type(e).__name__}: {e}"}
    
    feat = extract_features(y, sr=16000)
    probs = infer_probs(feat)  # Use smart inference function
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
        "model_mode": "AI" if MODEL_LOADED else "Demo",
        "classes_used": "20_javanese_aksara"
    }

@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Batch prediction for multiple audio files"""
    results = []
    
    for file in files:
        raw = await file.read()
        try:
            y = to_wav16k_mono_bytes(raw)
            feat = extract_features(y, sr=16000)
            probs = infer_probs(feat)
            
            pred_idx = int(np.argmax(probs))
            pred_label = CLASSES[pred_idx]
            conf = float(probs[pred_idx])
            
            results.append({
                "filename": file.filename,
                "pred": pred_label,
                "conf": round(conf, 3),
                "success": True
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {"results": results, "total": len(files)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
