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
const folderPicker = document.getElementById('folderPicker');
const openFolderBtn = document.getElementById('openFolderBtn');
const folderInfo = document.getElementById('folderInfo');
const imageCount = document.getElementById('imageCount');
const navigationControls = document.getElementById('navigationControls');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const imagePosition = document.getElementById('imagePosition');
const thresholdInput = document.getElementById('threshold');
const thresholdValue = document.getElementById('thresholdValue');
const predictBtn = document.getElementById('predictBtn');
const displayImageOriginal = document.getElementById('displayImageOriginal');
const displayImageResult = document.getElementById('displayImageResult');
const imageContainerOriginal = document.getElementById('imageContainerOriginal');
const imageContainerResult = document.getElementById('imageContainerResult');
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

    // Open folder button (local folder picker)
    openFolderBtn.addEventListener('click', () => folderPicker.click());
    folderPicker.addEventListener('change', handleLocalFolderSelect);

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
        handleFile(file, true);
    }
}

// Handle single file
async function handleFile(file, isSingle = false) {
    if (!file.type.startsWith('image/')) {
        showToast('请选择图片文件！', 'error');
        return;
    }
    // Reset folder mode
    if (isSingle) {
        imageList = [];
        currentFolderPath = '';
        navigationControls.style.display = 'none';
        folderInfo.style.display = 'none';
    }
    // Display original image (left)
    const reader = new FileReader();
    reader.onload = (e) => {
        displayImageOriginal.src = e.target.result;
        displayImageOriginal.style.display = 'block';
        imageContainerOriginal.querySelector('.placeholder').style.display = 'none';
        // Clear result image
        displayImageResult.style.display = 'none';
        imageContainerResult.querySelector('.placeholder').style.display = 'block';
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
// Handle local folder selection (HTML5 directory picker)
function handleLocalFolderSelect(e) {
    const files = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
    if (!files.length) {
        showToast('文件夹中没有图片！', 'error');
        return;
    }
    // Sort by name
    files.sort((a, b) => a.name.localeCompare(b.name, 'zh-CN', {numeric: true}));
    imageList = files;
    currentFolderPath = '';
    currentImageIndex = 0;
    imageCount.textContent = files.length;
    folderInfo.style.display = 'block';
    navigationControls.style.display = 'block';
    showImageAtIndexLocal(0);
    showToast(`成功加载 ${files.length} 张图片！`, 'success');
}

// Show image at index for local folder
async function showImageAtIndexLocal(index) {
    if (index < 0 || index >= imageList.length) return;
    currentImageIndex = index;
    const file = imageList[index];
    imagePosition.textContent = `${index + 1} / ${imageList.length}`;
    prevBtn.disabled = (index === 0);
    nextBtn.disabled = (index === imageList.length - 1);
    
    // 禁用预测按钮直到上传完成
    predictBtn.disabled = true;
    
    // Show original image
    const reader = new FileReader();
    reader.onload = (e) => {
        displayImageOriginal.src = e.target.result;
        displayImageOriginal.style.display = 'block';
        imageContainerOriginal.querySelector('.placeholder').style.display = 'none';
        // Clear result image
        displayImageResult.style.display = 'none';
        imageContainerResult.querySelector('.placeholder').style.display = 'flex';
        currentImageName.textContent = file.name;
    };
    reader.readAsDataURL(file);
    
    // Upload to server for prediction - 等待上传完成
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
            predictBtn.disabled = false;
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

// Show image at index (server folder mode, fallback)
async function showImageAtIndex(index) {
    if (Array.isArray(imageList) && imageList.length && imageList[0] instanceof File) {
        showImageAtIndexLocal(index);
        return;
    }
    if (index < 0 || index >= imageList.length) return;
    currentImageIndex = index;
    const imageName = imageList[index];
    currentImagePath = `${currentFolderPath}\\${imageName}`;
    imagePosition.textContent = `${index + 1} / ${imageList.length}`;
    prevBtn.disabled = (index === 0);
    nextBtn.disabled = (index === imageList.length - 1);
    try {
        const response = await fetch(`/api/get_image?path=${encodeURIComponent(currentImagePath)}`);
        const data = await response.json();
        if (data.success) {
            displayImageOriginal.src = data.image;
            displayImageOriginal.style.display = 'block';
            imageContainerOriginal.querySelector('.placeholder').style.display = 'none';
            // Clear result image
            displayImageResult.style.display = 'none';
            imageContainerResult.querySelector('.placeholder').style.display = 'block';
            currentImageName.textContent = imageName;
            predictBtn.disabled = false;
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
            // Show result image (right)
            displayImageResult.src = data.image;
            displayImageResult.style.display = 'block';
            imageContainerResult.querySelector('.placeholder').style.display = 'none';
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
    // 分类数量
    const riceNeskBlastCount = document.getElementById('riceNeskBlastCount');
    const riceBlastCount = document.getElementById('riceBlastCount');
    riceNeskBlastCount.textContent = data.results ? (data.results.rice_nesk_blast_count || 0) : 0;
    riceBlastCount.textContent = data.results ? (data.results.rice_blast_count || 0) : 0;
    
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
