# Model Directory

## Final Production Model

- **javanese_enhanced_retrain.h5** - Final enhanced CNN model
  - Accuracy: 94.67%
  - Parameters: 210,164
  - Architecture: CNN with GlobalAveragePooling2D and regularization
  - Input: Mel-spectrogram (87 time frames, 64 mel bins)
  
- **label_encoder_retrain.pkl** - Label encoder for 20 Javanese aksara classes

## Usage

```python
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model('javanese_enhanced_retrain.h5')

# Load label encoder  
with open('label_encoder_retrain.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
```

## Supported Classes

20 Javanese aksara: ha, na, ca, ra, ka, da, ta, sa, wa, la, pa, dha, ja, ya, nya, ma, ga, ba, tha, nga
