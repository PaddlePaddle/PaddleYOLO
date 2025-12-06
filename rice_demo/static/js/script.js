/**
 * Rice Blast Disease Detection System
 * JavaScript Functions for Interactive Features
 */

// Global variables
let currentImageIndex = 0;
let imageList = [];
let currentFolderPath = '';
let currentImagePath = '';

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const folderPath = document.getElementById('folderPath');
const loadFolderBtn = document.getElementById('loadFolderBtn');
const folderInfo = document.getElementById('folderInfo');
const imageCount = document.getElementById('imageCount');
const navigationControls = document.getElementById('navigationControls');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const imagePosition = document.getElementById('imagePosition');
const thresholdInput = document.getElementById('threshold');
const thresholdValue = document.getElementById('thresholdValue');
const predictBtn = document.getElementById('predictBtn');
const displayImage = document.getElementById('displayImage');
const imageContainer = document.getElementById('imageContainer');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsContent = document.getElementById('resultsContent');
const detectionCount = document.getElementById('detectionCount');
const currentImageName = document.getElementById('currentImageName');
const detectionList = document.getElementById('detectionList');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', handleFileUpload);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--rice-gold)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--rice-green)';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--rice-green)';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Load folder button
    loadFolderBtn.addEventListener('click', handleLoadFolder);
    
    // Navigation buttons
    prevBtn.addEventListener('click', showPreviousImage);
    nextBtn.addEventListener('click', showNextImage);
    
    // Threshold slider
    thresholdInput.addEventListener('input', (e) => {
        thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
    
    // Predict button
    predictBtn.addEventListener('click', handlePredict);
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (imageList.length > 0) {
            if (e.key === 'ArrowLeft') showPreviousImage();
            if (e.key === 'ArrowRight') showNextImage();
            if (e.key === 'Enter' && !predictBtn.disabled) handlePredict();
        }
    });
}

// Handle file upload
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Handle single file
async function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('请选择图片文件！', 'error');
        return;
    }
    
    // Reset folder mode
    imageList = [];
    currentFolderPath = '';
    navigationControls.style.display = 'none';
    folderInfo.style.display = 'none';
    
    // Display image
    const reader = new FileReader();
    reader.onload = (e) => {
        displayImage.src = e.target.result;
        displayImage.style.display = 'block';
        imageContainer.querySelector('.placeholder').style.display = 'none';
        currentImageName.textContent = file.name;
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    
    // Upload to server
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success) {
            currentImagePath = data.path;
            showToast('图片上传成功！', 'success');
        } else {
            showToast('图片上传失败：' + data.error, 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showToast('图片上传失败！', 'error');
    }
}

// Handle load folder
async function handleLoadFolder() {
    const path = folderPath.value.trim();
    if (!path) {
        showToast('请输入文件夹路径！', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/load_folder', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ folder_path: path })
        });
        
        const data = await response.json();
        if (data.success) {
            imageList = data.images;
            currentFolderPath = data.folder_path;
            currentImageIndex = 0;
            
            // Show folder info
            imageCount.textContent = data.total;
            folderInfo.style.display = 'block';
            
            if (imageList.length > 0) {
                navigationControls.style.display = 'block';
                showImageAtIndex(0);
                showToast(`成功加载 ${data.total} 张图片！`, 'success');
            } else {
                showToast('文件夹中没有找到图片！', 'error');
            }
        } else {
            showToast('加载文件夹失败：' + data.error, 'error');
        }
    } catch (error) {
        console.error('Load folder error:', error);
        showToast('加载文件夹失败！', 'error');
    }
}

// Show image at index
async function showImageAtIndex(index) {
    if (index < 0 || index >= imageList.length) return;
    
    currentImageIndex = index;
    const imageName = imageList[index];
    currentImagePath = `${currentFolderPath}\\${imageName}`;
    
    // Update position
    imagePosition.textContent = `${index + 1} / ${imageList.length}`;
    
    // Update buttons
    prevBtn.disabled = (index === 0);
    nextBtn.disabled = (index === imageList.length - 1);
    
    // Load image
    try {
        const response = await fetch(`/api/get_image?path=${encodeURIComponent(currentImagePath)}`);
        const data = await response.json();
        
        if (data.success) {
            displayImage.src = data.image;
            displayImage.style.display = 'block';
            imageContainer.querySelector('.placeholder').style.display = 'none';
            currentImageName.textContent = imageName;
            predictBtn.disabled = false;
            
            // Hide previous results
            resultsContent.style.display = 'none';
            resultsContent.previousElementSibling.style.display = 'block';
        } else {
            showToast('加载图片失败：' + data.error, 'error');
        }
    } catch (error) {
        console.error('Load image error:', error);
        showToast('加载图片失败！', 'error');
    }
}

// Show previous image
function showPreviousImage() {
    if (currentImageIndex > 0) {
        showImageAtIndex(currentImageIndex - 1);
    }
}

// Show next image
function showNextImage() {
    if (currentImageIndex < imageList.length - 1) {
        showImageAtIndex(currentImageIndex + 1);
    }
}

// Handle predict
async function handlePredict() {
    if (!currentImagePath) {
        showToast('请先选择图片！', 'error');
        return;
    }
    
    // Show loading
    loadingSpinner.style.display = 'flex';
    predictBtn.disabled = true;
    
    const threshold = parseFloat(thresholdInput.value);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_path: currentImagePath,
                threshold: threshold
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update image with bboxes
            displayImage.src = data.image;
            
            // Update results
            updateResults(data);
            
            showToast(`检测完成！发现 ${data.detections} 个病斑`, 'success');
        } else {
            showToast('检测失败：' + data.error, 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showToast('检测失败！', 'error');
    } finally {
        loadingSpinner.style.display = 'none';
        predictBtn.disabled = false;
    }
}

// Update results display
function updateResults(data) {
    // Show results section
    resultsContent.style.display = 'block';
    resultsContent.previousElementSibling.style.display = 'none';
    
    // Update detection count
    detectionCount.textContent = data.detections;
    
    // Clear previous detection list
    detectionList.innerHTML = '';
    
    // Add detection items
    if (data.results && data.results.bboxes) {
        const { bboxes, scores, class_ids, class_names } = data.results;
        
        bboxes.forEach((bbox, idx) => {
            const score = scores[idx];
            const className = class_names[class_ids[idx]] || '未知';
            
            const item = document.createElement('div');
            item.className = 'detection-item';
            item.innerHTML = `
                <div class="detection-item-header">
                    <span class="detection-item-name">
                        <i class="fas fa-virus"></i> ${className}
                    </span>
                    <span class="detection-item-score">${(score * 100).toFixed(1)}%</span>
                </div>
                <div class="detection-item-bbox">
                    <i class="fas fa-map-marker-alt"></i> 
                    位置: [${bbox[0].toFixed(0)}, ${bbox[1].toFixed(0)}, ${bbox[2].toFixed(0)}, ${bbox[3].toFixed(0)}]
                </div>
            `;
            detectionList.appendChild(item);
        });
    }
    
    if (data.detections === 0) {
        detectionList.innerHTML = '<p style="text-align: center; color: var(--rice-green); padding: 20px;"><i class="fas fa-check-circle"></i> 未检测到病斑，水稻健康！</p>';
    }
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.className = 'toast';
    }, 3000);
}

// Utility function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}
