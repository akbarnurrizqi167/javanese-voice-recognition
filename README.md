# Javanese Voice Recognition Demo

A simple voice recognition system for 20 Javanese aksara characters.

## Quick Start

### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate jv-voice
```

### 2. Run Gradio Interface (Recommended)
```bash
python enhanced_gradio_app.py
```

### 3. Or Run FastAPI Backend
```bash
cd backend && python main.py
```

Then open `frontend/index.html` in your browser.

## Project Structure

```
javanese-voice-demo/
├── backend/                # FastAPI backend
│   ├── main.py            # Main API server
│   └── main_tts_backup.py # Backup version
├── frontend/              # Web frontend files
│   ├── index.html         # Main interface
│   ├── test_simple.html   # Simple test page
│   └── enhanced_test.html # Enhanced test page
├── models/                # Trained models
│   ├── javanese_enhanced_retrain.h5    # Final model (94.67% accuracy)
│   ├── label_encoder_retrain.pkl       # Label encoder
│   └── README.md                       # Model documentation
├── enhanced_gradio_app.py # Enhanced Gradio interface
├── gradio_app.py         # Basic Gradio interface
├── enhanced_retrain.py   # Model retraining script
├── environment.yml       # Conda environment
└── README.md            # This file
```

## Features

- **20 Javanese Aksara**: ha, na, ca, ra, ka, da, ta, sa, wa, la, pa, dha, ja, ya, nya, ma, ga, ba, tha, nga
- **Enhanced CNN Model**: 94.67% accuracy with improved architecture
- **Real-time Prediction**: FastAPI backend with web interface
- **Multiple Interfaces**: Gradio web app and HTML frontend

## Model Performance

- **Final Model**: `javanese_enhanced_retrain.h5`
- **Accuracy**: 94.67%
- **Architecture**: Enhanced CNN with regularization
- **Input**: Mel-spectrogram features (22050 Hz, 2 seconds)

## API Endpoints

- `GET /health` - Service health check
- `GET /classes` - Available aksara classes  
- `POST /predict` - Single audio prediction
- `POST /batch_predict` - Batch audio prediction

## Audio Format

- **Recording**: WebM/OGG/MP4 from browser
- **Processing**: Automatic conversion to WAV 16kHz mono
- **Duration**: 2 seconds (padded/truncated)

---

**Note**: For production deployment, use the files in `/Users/user/javanese-api-deploy/`
