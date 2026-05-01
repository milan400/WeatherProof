import os
import sys
import uuid
import json
import threading
import logging
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable werkzeug request logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# Load configuration
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load config.json: {e}")
    sys.exit(1)

app.config['UPLOAD_FOLDER'] = config['upload_folder']
app.config['OUTPUT_FOLDER'] = config['output_folder']
app.config['MAX_CONTENT_LENGTH'] = config['max_file_size'] * 1024 * 1024

# Ensure directories exist
for directory in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], 'static/css', 'static/js']:
    os.makedirs(directory, exist_ok=True)

logger.info(f"Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
logger.info(f"Output folder: {os.path.abspath(app.config['OUTPUT_FOLDER'])}")

# Lazy import processing module
enhance_video = None

def get_enhance_video():
    global enhance_video
    if enhance_video is None:
        logger.info("Loading processing module...")
        from processing import enhance_video as ev
        enhance_video = ev
        logger.info("Processing module loaded successfully")
    return enhance_video

# Store processing tasks
processing_tasks = {}
tasks_lock = threading.Lock()

class ProcessingTask:
    def __init__(self, task_id, original_filename):
        self.task_id = task_id
        self.original_filename = original_filename
        self.progress = 0
        self.status = 'starting'
        self.message = 'Initializing...'
        self.input_path = None
        self.output_path = None
        self.error = None
        self.created_at = datetime.now()
        
    def update_progress(self, progress, message):
        self.progress = min(progress, 100)
        self.message = message
        
    def mark_completed(self):
        self.status = 'completed'
        self.progress = 100
        
    def mark_error(self, error):
        self.status = 'error'
        self.error = str(error)
        self.message = f'Error: {str(error)}'
        
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'progress': self.progress,
            'status': self.status,
            'message': self.message,
            'original_filename': self.original_filename,
            'input_url': url_for('static', filename=f'uploads/{os.path.basename(self.input_path)}') if self.input_path and os.path.exists(self.input_path) else None,
            'output_url': url_for('static', filename=f'outputs/{os.path.basename(self.output_path)}') if self.output_path and os.path.exists(self.output_path) else None,
            'error': self.error
        }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config['allowed_extensions']

def process_video_async(task):
    try:
        task.status = 'processing'
        task.message = 'Starting video enhancement...'
        
        def progress_callback(progress, message):
            task.update_progress(progress, message)
        
        enhance_func = get_enhance_video()
        enhance_func(task.input_path, task.output_path, progress_callback)
        
        task.mark_completed()
        
    except Exception as e:
        task.mark_error(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed'}), 400
        
        task_id = str(uuid.uuid4())[:8]
        original_filename = secure_filename(file.filename)
        input_filename = f"{task_id}_{original_filename}"
        output_filename = f"{task_id}_enhanced.mp4"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        file.save(input_path)
        logger.info(f"Video saved: {input_filename}")
        
        task = ProcessingTask(task_id, original_filename)
        task.input_path = input_path
        task.output_path = output_path
        
        with tasks_lock:
            processing_tasks[task_id] = task
        
        thread = threading.Thread(target=process_video_async, args=(task,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'message': 'Video uploaded. Processing started.'
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    with tasks_lock:
        task = processing_tasks.get(task_id)
    
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(task.to_dict())

@app.route('/download/<task_id>')
def download_video(task_id):
    with tasks_lock:
        task = processing_tasks.get(task_id)
    
    if not task or task.status != 'completed':
        return jsonify({'error': 'Video not ready'}), 404
    
    return send_file(task.output_path, as_attachment=True, download_name='enhanced_video.mp4')

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Video Enhancement Server Starting...")
    logger.info(f"Server: http://localhost:5000")
    logger.info("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)