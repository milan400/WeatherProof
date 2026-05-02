# WeatherProof: Gemma 4 Edge AI for Clear Vision Anywhere

Real-time video enhancement that removes snow, rain, fog, haze, and low-light — powered by Google Gemma 4 Multimodal VLM, running entirely on edge devices.

## How It Works

Gemma 4 analyzes video frames, detects adverse weather conditions, and selects the right AI model to restore clarity — no cloud needed.

Pipeline: Input Video → Gemma 4 (detection) → CIDNet / Attention UNet (restoration) → Enhanced Video

Smart Detection: VLM is not called on every frame. A fast histogram comparison reuses the previous plan until the scene changes significantly. Additionally, a periodic safety check forces a VLM call every 10 seconds to ensure detection is still accurate.

## Quick Setup

1. Install dependencies
   pip install -r requirements.txt

2. Download model weights (place in weights/ folder)
   Google Drive link: https://drive.google.com/drive/folders/1xCY5pdf0pg-qKcT1Cdejzc5AnmKbWnil?usp=sharing

3. Download FFmpeg binaries (place in project root)
   Google Drive link: https://drive.google.com/drive/folders/1ZDbbrVfn-esy5jJJW8ql_9SRCqmh8AcJ?usp=sharing

4. Install Gemma 4 model
   ollama pull gemma4:e2b

5. Run
   python app.py
   Open http://localhost:5000

## Key Features

- Gemma 4 Multimodal VLM for intelligent degradation detection
- Removes rain, snow, fog, haze, and low-light
- Runs on edge devices (CPU-only mode available)
- Change-triggered detection + periodic safety check minimizes VLM calls
- Side-by-side original vs. enhanced video preview

## Tech Stack

| Component               | Technology               |
|--------------------------|---------------------------|
| Vision Intelligence      | Gemma 4 (E2B)             |
| Low-light Enhancement    | CIDNet (PyTorch)          |
| Weather Removal          | Attention UNet (TensorFlow)|
| Runtime                  | Ollama                    |
| Interface                | Flask + HTML/CSS/JS       |

## Applications

Autonomous vehicles, drones, security cameras, content creation, search & rescue.

## License

MIT
