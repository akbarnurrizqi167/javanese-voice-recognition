import numpy as np, gradio as gr, librosa

# 20 Aksara Jawa Lengkap
CLASSES = ["ha", "na", "ca", "ra", "ka", "da", "ta", "sa", "wa", "la", "pa", "dha", "ja", "ya", "nya", "ma", "ga", "ba", "tha", "nga"]

# Load trained model if available
try:
    import tensorflow as tf
    model_paths = [
        'models/javanese_20_model.h5',
        'best_javanese_model_20.h5',
        'best_javanese_model.h5'
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
        print("⚠️ No trained model found. Using demo mode.")
        
except:
    print("⚠️ TensorFlow not available. Using demo mode.")
    MODEL_LOADED = False
    model = None

def extract_features(y, sr=16000, target_len=1.0, n_mels=64):
    # trim/pad ke 1 detik
    T = int(target_len*sr)
    if len(y) > T:
        y = y[:T]
    else:
        y = np.pad(y, (0, T-len(y)))

    # === Mel-spectrogram manual ===
    n_fft = 400
    hop = 160
    # STFT -> power spec
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop, center=False))**2
    # Mel filterbank
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    M = np.dot(mel_filter, S)

    # log + CMVN
    M = np.log(M + 1e-6)
    M = (M - M.mean()) / (M.std() + 1e-9)

    # [1, H, W, 1] buat CNN
    return M.astype(np.float32)[None, :, :, None]

def predict_fn(audio: tuple, target_label: str):
    if audio is None:
        return "Tidak ada audio", None, None
    
    sr, data = audio

    # pastikan float32 mono
    y = data.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
        sr = 16000

    feat = extract_features(y=y, sr=sr)

    if MODEL_LOADED:
        # Real model prediction
        predictions = model.predict(feat, verbose=0)
        probs = predictions[0]
    else:
        # Demo mode: random probabilities
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(len(CLASSES),))
        probs = np.exp(logits) / np.sum(np.exp(logits))

    pred_idx = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]
    conf = float(probs[pred_idx])

    if target_label in CLASSES:
        t_idx = CLASSES.index(target_label)
        target_conf = float(probs[t_idx])
        score = round(100 * target_conf, 1)
        
        if score >= 70:
            status = "✅ Benar"
        elif score >= 50:
            status = "⚠️ Hampir"
        else:
            status = "❌ Coba lagi"
            
        mode_indicator = "🤖 AI Model" if MODEL_LOADED else "🎲 Demo Mode"
        msg = f"{status} — Target: {target_label} | Pred: {pred_label} ({conf:.2f}) | Skor: {score}% | {mode_indicator}"
    else:
        mode_indicator = "🤖 AI Model" if MODEL_LOADED else "🎲 Demo Mode"
        msg = f"❌ Target '{target_label}' tidak didukung. Pilih dari: {', '.join(CLASSES)} | {mode_indicator}"

    top5_idx = np.argsort(-probs)[:5]
    top5 = {CLASSES[i]: float(probs[i]) for i in top5_idx}
    
    return msg, pred_label, top5

# Custom CSS
css = """
.gradio-container {
    background: linear-gradient(45deg, #1e3c72, #2a5298);
    color: white;
}
.gr-button {
    background: linear-gradient(45deg, #ff6b6b, #ee5a24);
    border: none;
    color: white;
    font-weight: bold;
}
"""

# Gradio Interface
with gr.Blocks(title="Javanese Aksara TTS Trainer", css=css) as demo:
    gr.Markdown("# 🎯 Javanese Aksara Voice Recognition (TTS Dataset)")
    gr.Markdown("### 🎵 20 Aksara Jawa: ha, na, ca, ra, ka, da, ta, sa, wa, la, pa, dha, ja, ya, nya, ma, ga, ba, tha, nga")
    
    if MODEL_LOADED:
        gr.Markdown("#### ✅ **Status: AI Model Aktif** - Menggunakan model yang sudah dilatih")
    else:
        gr.Markdown("#### ⚠️ **Status: Demo Mode** - Belum ada model terlatih, menggunakan prediksi acak")
        gr.Markdown("💡 **Untuk training**: Jalankan `python scripts/tts/generate_tts_dataset.py` untuk generate dataset")
    
    with gr.Row():
        with gr.Column(scale=1):
            target = gr.Dropdown(
                choices=CLASSES, 
                value="ha", 
                label="🎯 Target Aksara",
                info="Pilih aksara yang ingin dilatih"
            )
            audio_in = gr.Audio(
                sources=["microphone"], 
                type="numpy", 
                label="🎙️ Rekam Suara",
                info="Ucapkan aksara target dengan jelas (≤ 2 detik)"
            )
            btn = gr.Button("🎵 Nilai Pengucapan", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            out_msg = gr.Textbox(
                label="📊 Hasil Penilaian",
                info="Status dan skor pengucapan"
            )
            out_pred = gr.Textbox(
                label="🤖 Prediksi Model",
                info="Aksara yang diprediksi oleh model"
            )
            out_top5 = gr.JSON(
                label="📈 Top-5 Probabilitas",
                info="5 prediksi teratas dengan confidence score"
            )
    
    # Instructions
    with gr.Accordion("📖 Cara Penggunaan", open=False):
        gr.Markdown("""
        ### 🎯 Langkah-langkah:
        1. **Pilih Target**: Pilih aksara yang ingin dilatih dari dropdown
        2. **Rekam Suara**: Klik microphone dan ucapkan aksara dengan jelas
        3. **Evaluasi**: Klik tombol untuk mendapat skor dan feedback
        
        ### 🎵 Tips Pengucapan:
        - Ucapkan dengan **jelas** dan **konsisten**
        - Durasi ideal: **1-2 detik**
        - Lingkungan yang **tenang**
        - Volume suara **stabil**
        
        ### 📊 Sistem Penilaian:
        - **✅ Benar** (≥70%): Pengucapan sangat baik
        - **⚠️ Hampir** (50-69%): Perlu sedikit perbaikan  
        - **❌ Coba Lagi** (<50%): Perlu latihan lebih
        """)
    
    # TTS Dataset Info
    with gr.Accordion("🎵 TTS Dataset Generation", open=False):
        gr.Markdown("""
        ### 🤖 Generate Dataset dengan TTS:
        ```bash
        # Install TTS dependencies
        pip install gtts pyttsx3 pydub
        
        # Test TTS engines
        python scripts/tts/test_tts.py
        
        # Generate TTS dataset (500 samples per class)
        python scripts/tts/generate_tts_dataset.py
        ```
        
        ### 📊 TTS Dataset akan menghasilkan:
        - **20 classes** × **50 samples** = **1000 audio files**
        - **Multiple TTS engines** untuk variasi
        - **Text variations** untuk diversity
        - **Metadata JSON** untuk tracking
        """)
    
    btn.click(fn=predict_fn, inputs=[audio_in, target], outputs=[out_msg, out_pred, out_top5])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7863, share=False)
