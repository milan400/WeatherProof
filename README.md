# WeatherProof: Gemma 4 Edge AI for Clear Vision Anywhere

Real‑time video enhancement that removes snow, rain, fog, haze and low‑light — powered by Google Gemma 4 Multimodal VLM, running entirely on edge devices.

## Project Structure

weatherproof/
├── app.py
├── processing.py
├── config.json
├── requirements.txt
├── run.bat
├── ffmpeg.exe
├── ffprobe.exe
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   ├── uploads/
│   └── outputs/
└── weights/
    ├── LOLv2_real/
    │   └── w_perc.pth
    ├── dehaze.keras
    ├── desnow.keras
    └── derain.keras
## How It Works

Input Video → Gemma 4 (detection) → CIDNet / Attention UNet (restoration) → Enhanced Video

Smart Detection: VLM is not called on every frame. A fast histogram comparison reuses the previous plan until the scene changes. A safety check forces a fresh VLM call every 10 seconds.

## Quick Setup

1. Install dependencies  
   pip install -r requirements.txt

2. Download model weights → place in weights/  
   https://drive.google.com/drive/folders/1xCY5pdf0pg-qKcT1Cdejzc5AnmKbWnil

3. Download FFmpeg → place in project root  
   https://drive.google.com/drive/folders/1ZDbbrVfn-esy5jJJW8ql_9SRCqmh8AcJ

4. Install Gemma 4 model  
   ollama pull gemma4:e2b

5. Launch  
   python app.py  
   Open http://localhost:5000

## Key Features

- Gemma 4 Multimodal VLM for intelligent degradation detection
- Removes rain, snow, fog, haze, and low-light
- Change‑triggered + periodic safety check minimises VLM calls
- Side‑by‑side original vs. enhanced video preview

## Tech Stack

 Vision Intelligence   → Gemma 4 (E2B)  
 Low‑light Enhancement  → CIDNet (PyTorch)  
 Weather Removal        → Attention UNet (TensorFlow)  
 Runtime                → Ollama  
 Interface              → Flask + HTML/CSS/JS  

## License

MIT
