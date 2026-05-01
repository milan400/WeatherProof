import os
import sys
import subprocess
import shutil
import tempfile
import json
import re
import logging
import time
import atexit
import requests
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ===== OLLAMA AUTO-STARTUP =====
ollama_process = None
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e2b"

def cleanup_ollama():
    """Cleanup function to stop Ollama when script exits"""
    global ollama_process
    if ollama_process:
        logger.info("Shutting down Ollama server...")
        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(ollama_process.pid)], 
                             capture_output=True)
            else:
                ollama_process.terminate()
                ollama_process.wait(timeout=5)
        except:
            pass
        ollama_process = None

atexit.register(cleanup_ollama)

def is_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_ollama_server():
    """Start Ollama server"""
    global ollama_process
    
    if is_ollama_running():
        logger.info("Ollama server is already running")
        return True
    
    logger.info("Starting Ollama server...")
    
    try:
        if sys.platform == 'win32':
            ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
        
        logger.info("Waiting for Ollama server...")
        for i in range(30):
            if is_ollama_running():
                logger.info(f"Ollama server started (took {i+1}s)")
                return True
            time.sleep(1)
        
        logger.error("Ollama server failed to start")
        return False
        
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False

def ensure_model_available(model_name=OLLAMA_MODEL):
    """Check if model is available, pull if not"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            if model_name in model_names:
                logger.info(f"Model '{model_name}' is available")
                return True
        
        logger.info(f"Pulling model '{model_name}'...")
        import ollama as ollama_lib
        
        for progress in ollama_lib.pull(model_name, stream=True):
            if 'status' in progress:
                logger.info(f"Pull: {progress['status']}")
        
        logger.info(f"Model '{model_name}' pulled successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model error: {e}")
        return False

def initialize_ollama():
    """Initialize Ollama - start server and ensure model"""
    logger.info("=" * 60)
    logger.info("Initializing Ollama System...")
    logger.info("=" * 60)
    
    if not start_ollama_server():
        logger.warning("Ollama not available - using basic mode")
        return False
    
    if not ensure_model_available(OLLAMA_MODEL):
        logger.warning(f"Model '{OLLAMA_MODEL}' not available")
        return False
    
    logger.info("Ollama system ready")
    return True

_ollama_initialized = False

def get_ollama():
    """Lazy initialization of Ollama"""
    global _ollama_initialized
    if not _ollama_initialized:
        _ollama_initialized = initialize_ollama()
    return _ollama_initialized

# ===== FFMPEG PATH FIX =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = os.path.join(BASE_DIR, 'ffmpeg.exe')
FFPROBE_PATH = os.path.join(BASE_DIR, 'ffprobe.exe')

if not os.path.exists(FFMPEG_PATH):
    FFMPEG_PATH = r'C:\ffmpeg\bin\ffmpeg.exe'
    FFPROBE_PATH = r'C:\ffmpeg\bin\ffprobe.exe'
    
    if not os.path.exists(FFMPEG_PATH):
        FFMPEG_PATH = 'ffmpeg'
        FFPROBE_PATH = 'ffprobe'

logger.info(f"FFmpeg: {FFMPEG_PATH}")
logger.info(f"FFprobe: {FFPROBE_PATH}")

FFMPEG_ENV = os.environ.copy()
if 'CONDA_PREFIX' in FFMPEG_ENV:
    conda_bin = os.path.join(FFMPEG_ENV['CONDA_PREFIX'], 'Library', 'bin')
    paths = FFMPEG_ENV.get('PATH', '').split(os.pathsep)
    paths = [p for p in paths if conda_bin not in p]
    gtk_path = r'C:\Program Files\GTK3-Runtime Win64\bin'
    if os.path.exists(gtk_path):
        paths.insert(0, gtk_path)
    FFMPEG_ENV['PATH'] = os.pathsep.join(paths)

# ===== IMPORTS =====
import cv2
import tensorflow as tf
import ollama
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange

logger.info(f"OpenCV: {cv2.__version__}")
logger.info(f"TensorFlow: {tf.__version__}")
logger.info(f"PyTorch: {torch.__version__}")

# ===== CONFIG =====
with open('config.json', 'r') as f:
    config = json.load(f)

IMG_SIZE = config['img_size']
CONF_THRESH = config['conf_thresh']
PROCESSING_ORDER = config['processing_order']
CHANGE_THRESH = config['change_thresh']
FORCE_VLM_INTERVAL_SEC = config['force_vlm_interval_sec']
CRF = config['crf']
CIDNET_PATH = config['cidnet_path']
WEIGHT_DIR = config['weight_dir']

DEGRADATION_TO_MODEL = {
    "low_light": "delight", "haze": "dehaze", "fog": "dehaze",
    "snow": "desnow", "rain": "derain", "clean": None
}

# ===== CIDNet ARCHITECTURE =====
pi = 3.141592653589793

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0
        
    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ((img[:,0]-img[:,1]) / (value - img_min + eps))[img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ((img[:,2]-img[:,0]) / (value - img_min + eps))[img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps))[img[:,0]==value]) % 6
        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0
        saturation = (value - img_min) / (value + eps)
        saturation[value==0] = 0
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        k = self.density_k
        self.this_k = k.item()
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I], dim=1)
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H, V, I = img[:,0,:,:], img[:,1,:,:], img[:,2,:,:]
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        h = torch.atan2(V + eps, H + eps) / (2*pi)
        h = h % 1
        s = torch.sqrt(H**2 + V**2 + eps)
        if self.gated:
            s = s * self.alpha_s
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class NormDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=0.5, use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        
    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
        return x

class NormUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch*2, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
            
    def forward(self, x, y):
        x = self.up_scale(x)
        x = torch.cat([x, y], dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x

class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.Tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x

class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x

class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x)) 
        return x

class CIDNet(nn.Module):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False):
        super(CIDNet, self).__init__()
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False))
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False))
        
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False))
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False))
        
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()
        
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)
        return output_rgb

# ===== UNIVERSAL STABLE LOSS =====
class UniversalStableLoss(tf.keras.losses.Loss):
    def __init__(self, l1_weight=0.05, ssim_weight=0.1, grad_weight=0.05, name='universal_stable', **kwargs):
        super().__init__(name=name, **kwargs)
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight

    def call(self, y_true, y_pred):
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=11))
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        grad_loss = tf.reduce_mean(tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred))
        return (self.l1_weight * l1_loss +
                self.ssim_weight * ssim_loss +
                self.grad_weight * grad_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'l1_weight': self.l1_weight,
            'ssim_weight': self.ssim_weight,
            'grad_weight': self.grad_weight
        })
        return config

# ===== VLM + CHANGE DETECTOR =====
def query_gpt_llava(image_input):
    temp_file = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(temp_fd)
        image_input.convert("RGB").save(temp_path, "JPEG", quality=85)
        image_path = temp_path
        temp_file = temp_path

        prompt = (
            "<start_of_turn>user\n"
            "You are a degradation classifier for autonomous driving cameras.\n"
            "List ALL visual degradations present.\n"
            "Valid labels: ['rain', 'snow', 'haze', 'fog', 'low_light', 'clean'].\n"
            "Respond ONLY with JSON: {\"degradations\": [\"<label1>\", \"<label2>\"], \"confidences\": [<0-1>, <0-1>]}\n"
            "<start_of_turn>model\n"
        )
        
        # 🔑 SAME AS KAGGLE - Let Ollama decide GPU/CPU
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
            options={"temperature": 0.0}  # No gpu_layers, no num_threads
        )
        return response["message"]["content"]
        
    except Exception as e:
        logger.error(f"VLM error: {e}")
        return '{"degradations": ["clean"], "confidences": [0.0]}'
    finally:
        if temp_file and os.path.exists(temp_file):
            try: os.remove(temp_file)
            except: pass

def parse_vlm_json_multi(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match: return {"degradations": ["clean"], "confidences": [1.0]}
        data = json.loads(match.group())
        degs = data.get("degradations", ["clean"])
        confs = data.get("confidences", [0.0]*len(degs))
        if len(confs)!= len(degs): confs = [0.5] * len(degs)
        return {"degradations": degs, "confidences": [float(c) for c in confs]}
    except:
        return {"degradations": ["clean"], "confidences": [0.0]}

def frame_diff_score(img1_pil, img2_pil):
    """Fast V-channel hist distance on 64x64"""
    img1 = cv2.cvtColor(np.array(img1_pil.resize((64, 64))), cv2.COLOR_RGB2HSV)[:, :, 2]
    img2 = cv2.cvtColor(np.array(img2_pil.resize((64, 64))), cv2.COLOR_RGB2HSV)[:, :, 2]
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    h1 = cv2.calcHist([img1], [0], None, [32], [0, 256])
    h2 = cv2.calcHist([img2], [0], None, [32], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

# ===== RESTORATION MODELS =====
def load_restoration_models(weight_dir=WEIGHT_DIR):
    """Load restoration models"""
    models = {}
    
    if os.path.exists(CIDNET_PATH):
        try:
            logger.info("Loading CIDNet...")
            cidnet = CIDNet()
            cidnet.load_state_dict(torch.load(CIDNET_PATH, map_location='cpu'))
            cidnet.trans.gated = True
            cidnet.trans.gated2 = True
            cidnet.eval()
            models["delight"] = cidnet
            logger.info("CIDNet loaded")
        except Exception as e:
            logger.warning(f"CIDNet failed: {e}")
    else:
        logger.warning(f"CIDNet weights not found: {CIDNET_PATH}")
    
    for name in set(DEGRADATION_TO_MODEL.values()):
        if name is None or name == "delight":
            continue
        path = f"{weight_dir}/{name}.keras"
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(
                    path,
                    custom_objects={'UniversalStableLoss': UniversalStableLoss},
                    compile=False
                )
                logger.info(f"Loaded: {name}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
    
    logger.info(f"Loaded {len(models)} models total")
    return models

def preprocess_for_unet(pil_img):
    arr = np.array(pil_img.convert('RGB'), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def postprocess_from_unet(pred):
    pred = np.clip(pred[0], 0, 1)
    pred = (pred * 255).astype(np.uint8)
    return Image.fromarray(pred)

def cidnet_filtering(pil_img, model, gamma=1.0, alpha_s=1.0, alpha_i=1.0):
    torch.set_grad_enabled(False)
    
    pil2tensor = transforms.Compose([transforms.ToTensor()])
    input_tensor = pil2tensor(pil_img)
    factor = 8
    h, w = input_tensor.shape[1], input_tensor.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_tensor = F.pad(input_tensor.unsqueeze(0), (0, padw, 0, padh), 'reflect')

    with torch.no_grad():
        model.trans.alpha_s = alpha_s
        model.trans.alpha = alpha_i
        output = model(input_tensor ** gamma)

    output = torch.clamp(output, 0, 1)
    output = output[:, :, :h, :w]
    enhanced_img = transforms.ToPILImage()(output.squeeze(0))
    
    return enhanced_img

def apply_test_enhancement(pil_image):
    """Apply visible enhancement for testing when models aren't available"""
    img = pil_image.copy()
    
    arr = np.array(img)
    brightness = np.mean(arr)
    
    if brightness < 85:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.6)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
    elif brightness > 170:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
    else:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.15)
    
    return img

# ===== CHANGE-TRIGGERED ENHANCER =====
class ChangeTriggeredEnhancer:
    def __init__(self, weight_dir=WEIGHT_DIR, conf_thresh=CONF_THRESH):
        self.restoration_models = load_restoration_models(weight_dir)
        self.conf_thresh = conf_thresh
        self.current_plan = []
        self.current_detections = ['clean']
        self.last_vlm_img = None
        self.last_vlm_frame_idx = -1e9
        self.vlm_call_count = 0
        self.vlm_available = get_ollama()

    def _build_plan(self, degs, confs):
        plan = []
        for deg, conf in zip(degs, confs):
            if conf < self.conf_thresh: continue
            model_key = DEGRADATION_TO_MODEL.get(deg)
            if model_key and model_key not in plan:
                plan.append(model_key)
        return [m for m in PROCESSING_ORDER if m in plan]

    def _should_run_vlm(self, frame_idx, fps, current_img):
        if not self.vlm_available:
            return False, "vlm_unavailable"
        if self.last_vlm_img is None:
            return True, "first_frame"
        if (frame_idx - self.last_vlm_frame_idx) >= int(fps * FORCE_VLM_INTERVAL_SEC):
            return True, "timeout"
        diff = frame_diff_score(self.last_vlm_img, current_img)
        if diff > CHANGE_THRESH:
            return True, f"change_{diff:.3f}"
        return False, "reuse"

    def enhance(self, pil_image_256, frame_idx, fps):
        run_vlm, reason = self._should_run_vlm(frame_idx, fps, pil_image_256)

        if run_vlm:
            try:
                vlm_text = query_gpt_llava(pil_image_256)
                result = parse_vlm_json_multi(vlm_text)
                degs, confs = result["degradations"], result["confidences"]
                self.current_plan = self._build_plan(degs, confs)
                self.current_detections = degs
                logger.info(f"[VLM {self.vlm_call_count}] Detected: {degs} -> Plan: {self.current_plan}")
            except Exception as e:
                logger.warning(f"VLM failed: {e}")
            finally:
                self.last_vlm_img = pil_image_256.copy()
                self.last_vlm_frame_idx = frame_idx
                self.vlm_call_count += 1

        # If no models and no VLM, apply test enhancement for visible difference
        if not self.restoration_models:
            return apply_test_enhancement(pil_image_256), {
                "action": "test_mode",
                "applied": ["basic_enhancement"],
                "detections": self.current_detections
            }

        # If no plan, apply basic enhancement
        if not self.current_plan:
            return apply_test_enhancement(pil_image_256), {
                "action": "basic_enhance",
                "applied": ["auto_enhance"],
                "detections": self.current_detections
            }

        # Apply restoration models
        current_img = pil_image_256
        applied = []
        for model_key in self.current_plan:
            if model_key in self.restoration_models:
                try:
                    if model_key == "delight":
                        current_img = cidnet_filtering(current_img, self.restoration_models[model_key])
                    else:
                        inp = preprocess_for_unet(current_img)
                        pred = self.restoration_models[model_key].predict(inp, verbose=0)
                        current_img = postprocess_from_unet(pred)
                    applied.append(model_key)
                except Exception as e:
                    logger.error(f"Failed to apply {model_key}: {e}")

        return current_img, {
            "action": "enhanced",
            "applied": applied,
            "detections": self.current_detections
        }

# ===== MAIN VIDEO PROCESSING FUNCTION =====
def enhance_video(input_path, output_path, progress_callback=None):
    """
    Enhanced video processing function with detailed progress tracking
    """
    
    if progress_callback:
        progress_callback(0, "Initializing AI models...")
    
    get_ollama()
    
    enhancer = ChangeTriggeredEnhancer(weight_dir=WEIGHT_DIR)
    tmp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(tmp_dir, "frames")
    enhanced_dir = os.path.join(tmp_dir, "enhanced")
    audio_path = os.path.join(tmp_dir, "audio.aac")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)

    try:
        if progress_callback:
            progress_callback(5, "Analyzing video...")
            
        probe_cmd = [
            FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate",
            "-of", "json", input_path
        ]
        
        probe_out = subprocess.check_output(probe_cmd, env=FFMPEG_ENV, timeout=30).decode()
        stream_info = json.loads(probe_out)["streams"][0]
        fps_str = stream_info.get("avg_frame_rate") or stream_info.get("r_frame_rate")
        
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(fps_str)
        
        logger.info(f"Video FPS: {fps:.2f}")

        if progress_callback:
            progress_callback(10, f"Extracting frames at {fps:.1f} fps...")

        subprocess.run([
            FFMPEG_PATH, "-i", input_path, 
            "-vf", f"scale={IMG_SIZE}:{IMG_SIZE}",
            "-vsync", "0", "-qscale:v", "2", 
            f"{frames_dir}/%06d.png"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
           env=FFMPEG_ENV, timeout=300)

        if progress_callback:
            progress_callback(20, "Extracting audio...")

        try:
            subprocess.run([FFMPEG_PATH, "-i", input_path, "-vn", "-acodec", "copy", audio_path],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          env=FFMPEG_ENV, timeout=60)
            has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
        except:
            has_audio = False

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        total_frames = len(frame_files)
        logger.info(f"Processing {total_frames} frames...")
        
        if progress_callback:
            progress_callback(22, f"Starting enhancement of {total_frames} frames...")
        
        # Track processing stats for display
        vlm_calls = 0
        last_vlm_frame = 0
        detections_summary = {}
        processing_start_time = time.time()

        for idx, fname in enumerate(frame_files):
            in_path = os.path.join(frames_dir, fname)
            out_path = os.path.join(enhanced_dir, fname)
            img = Image.open(in_path)
            
            # Enhance the frame
            enhanced_img, meta = enhancer.enhance(img, idx, fps)
            enhanced_img.save(out_path, "PNG")
            
            # Track VLM calls
            if meta.get('action') in ['enhanced', 'cascade']:
                vlm_calls += 1
                last_vlm_frame = idx
                for det in meta.get('detections', []):
                    detections_summary[det] = detections_summary.get(det, 0) + 1
            
            # Update progress for EVERY frame
            if progress_callback:
                # Calculate progress (25% to 75% for frame processing)
                progress = 25 + (idx / total_frames) * 50
                
                # Calculate estimated time remaining
                elapsed = time.time() - processing_start_time
                if idx > 0:
                    time_per_frame = elapsed / (idx + 1)
                    remaining_frames = total_frames - (idx + 1)
                    eta_seconds = time_per_frame * remaining_frames
                    
                    if eta_seconds > 60:
                        eta_str = f"{eta_seconds/60:.1f} min remaining"
                    else:
                        eta_str = f"{eta_seconds:.0f}s remaining"
                else:
                    eta_str = "calculating..."
                
                # Build detailed status message
                action = meta.get('action', 'processing')
                applied = meta.get('applied', [])
                
                if applied:
                    applied_str = ", ".join(applied)
                    status_msg = f"Frame {idx+1}/{total_frames} | Applied: {applied_str} | {eta_str}"
                else:
                    status_msg = f"Frame {idx+1}/{total_frames} | Action: {action} | {eta_str}"
                
                progress_callback(progress, status_msg)
            
            # Log every 10 frames
            if idx % 10 == 0:
                logger.info(f"Frame {idx+1}/{total_frames} | {meta.get('action', 'unknown')} | {meta.get('applied', [])}")

        # Processing complete - build final summary
        processing_time = time.time() - processing_start_time
        
        if progress_callback:
            # Build summary message
            summary_parts = []
            if detections_summary:
                for deg, count in detections_summary.items():
                    summary_parts.append(f"{deg}: {count} frames")
            
            summary_msg = f"Processed {total_frames} frames in {processing_time:.1f}s"
            if summary_parts:
                summary_msg += f" | Detected: {', '.join(summary_parts)}"
            if vlm_calls > 0:
                summary_msg += f" | VLM calls: {vlm_calls}"
            
            progress_callback(80, summary_msg)
        
        logger.info(f"Processing complete: {total_frames} frames in {processing_time:.1f}s")
        logger.info(f"VLM calls: {vlm_calls}")

        if progress_callback:
            progress_callback(85, "Encoding final video...")

        encode_cmd = [FFMPEG_PATH, "-y", "-r", str(fps), "-i", f"{enhanced_dir}/%06d.png"]
        if has_audio:
            encode_cmd += ["-i", audio_path, "-c:a", "aac", "-shortest"]
        encode_cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF),
            "-s", f"{IMG_SIZE}x{IMG_SIZE}", "-r", str(fps), output_path
        ]
        
        subprocess.run(encode_cmd, check=True, env=FFMPEG_ENV, timeout=600)
        
        if progress_callback:
            progress_callback(100, f"Complete! {total_frames} frames enhanced in {processing_time:.1f}s")
            
        logger.info(f"Enhanced video saved: {output_path}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")
        raise
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except:
            pass