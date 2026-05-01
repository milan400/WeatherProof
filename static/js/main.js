let currentTask = null;
let statusInterval = null;

const uploadSection = document.getElementById('uploadSection');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const dropZone = document.getElementById('dropZone');
const videoInput = document.getElementById('videoInput');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const progressPercentage = document.getElementById('progressPercentage');
const originalVideo = document.getElementById('originalVideo');
const enhancedVideo = document.getElementById('enhancedVideo');
const downloadBtn = document.getElementById('downloadBtn');
const newVideoBtn = document.getElementById('newVideoBtn');
const retryBtn = document.getElementById('retryBtn');
const errorText = document.getElementById('errorText');

dropZone.addEventListener('click', () => videoInput.click());

videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        uploadVideo(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        uploadVideo(e.dataTransfer.files[0]);
    }
});

async function uploadVideo(file) {
    const formData = new FormData();
    formData.append('video', file);
    
    showSection('processing');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        currentTask = data.task_id;
        startStatusPolling();
        
    } catch (error) {
        showError(error.message);
    }
}

function startStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);
    
    statusInterval = setInterval(async () => {
        try {
            const response = await fetch(`/status/${currentTask}`);
            const data = await response.json();
            
            progressFill.style.width = `${data.progress}%`;
            progressPercentage.textContent = `${Math.round(data.progress)}%`;
            progressText.textContent = data.message;
            
            if (data.status === 'completed') {
                clearInterval(statusInterval);
                showResults(data.input_url, data.output_url);
            } else if (data.status === 'error') {
                clearInterval(statusInterval);
                showError(data.error);
            }
        } catch (error) {
            console.error('Status error:', error);
        }
    }, 1000);
}

function showResults(inputUrl, outputUrl) {
    originalVideo.src = inputUrl;
    enhancedVideo.src = outputUrl;
    
    showSection('results');
    
    downloadBtn.onclick = () => {
        window.location.href = `/download/${currentTask}`;
    };
}

function showError(message) {
    errorText.textContent = message;
    showSection('error');
}

function showSection(section) {
    uploadSection.style.display = 'none';
    processingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    if (section === 'upload') uploadSection.style.display = 'block';
    if (section === 'processing') processingSection.style.display = 'block';
    if (section === 'results') resultsSection.style.display = 'block';
    if (section === 'error') errorSection.style.display = 'block';
}

newVideoBtn.addEventListener('click', () => {
    originalVideo.src = '';
    enhancedVideo.src = '';
    currentTask = null;
    showSection('upload');
});

retryBtn.addEventListener('click', () => {
    showSection('upload');
});