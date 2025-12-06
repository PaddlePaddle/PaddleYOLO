# -*- coding: utf-8 -*-
"""
Rice Blast Disease Detection Web Application
Copyright (c) 2025
"""

import os
import sys
import json
import base64
from io import BytesIO
from pathlib import Path
import glob
import cv2

# Add parent path to sys.path
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_path)

from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.cli import merge_args
from ppdet.utils.logger import setup_logger

logger = setup_logger('rice_blast_app')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Global variables for model
trainer = None
cfg = None

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def init_model(config_path, weights_path, use_gpu=True):
    """Initialize the detection model"""
    global trainer, cfg
    
    logger.info(f"Loading config from: {config_path}")
    cfg = load_config(config_path)
    cfg.use_gpu = use_gpu
    cfg.weights = weights_path
    
    # Set device
    if use_gpu:
        place = paddle.set_device('gpu')
    else:
        place = paddle.set_device('cpu')
    
    # Build trainer
    trainer = Trainer(cfg, mode='test')
    
    # Load weights
    logger.info(f"Loading weights from: {weights_path}")
    trainer.load_weights(cfg.weights)
    
    logger.info("Model initialized successfully!")


def draw_bboxes_on_image(image_path, bboxes, scores, class_ids, class_names, threshold=0.5):
    """Draw bounding boxes on image"""
    # Read image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Color for rice blast disease (red-ish for disease detection)
    color = (255, 69, 0)  # Red-orange for disease
    
    # Draw each bbox
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        if score >= threshold:
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else str(int(class_id))
            label = f"{class_name}: {score:.2f}"
            
            # Draw label background
            text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font)
    
    return image


def predict_image(image_path, threshold=0.5):
    """Run prediction on a single image using the same pipeline as infer.py"""
    global trainer, cfg
    
    if trainer is None:
        raise ValueError("Model not initialized. Please call init_model first.")
    
    logger.info(f"Running prediction on {image_path}")
    
    try:
        # Get class names - use correct import path
        from ppdet.data.source.category import get_categories
        from ppdet.core.workspace import create
        
        anno_file = trainer.dataset.get_anno()
        clsid2catid, catid2name = get_categories(cfg.metric, anno_file=anno_file)
        class_names = [catid2name[i] if i in catid2name else f'class_{i}' for i in sorted(catid2name.keys())]
        
        if not class_names:
            class_names = ['rice_blast']  # Default class name
        
        # Use the same predict method as infer.py
        # Set images and create TestReader loader
        trainer.dataset.set_images([image_path])
        loader = create('TestReader')(trainer.dataset, 0)
        
        # Run inference using the same pipeline as Trainer.predict()
        trainer.model.eval()
        results = []
        
        for step_id, data in enumerate(loader):
            # Forward pass
            with paddle.no_grad():
                outs = trainer.model(data)
            
            # Get im_shape, scale_factor, im_id from data
            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, (list, tuple)):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            
            # Convert to numpy
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            
            results.append(outs)
        
        # Extract bboxes from results
        bboxes = []
        scores = []
        class_ids = []
        
        for outs in results:
            if 'bbox' in outs and len(outs['bbox']) > 0:
                # outs['bbox'] shape: [N, 6] where N is number of detections
                # Each row: [class_id, score, x1, y1, x2, y2]
                bbox_data = outs['bbox']
                bbox_num = int(outs['bbox_num'][0]) if 'bbox_num' in outs else len(bbox_data)
                
                logger.info(f"Raw bbox data shape: {bbox_data.shape if hasattr(bbox_data, 'shape') else 'unknown'}")
                logger.info(f"Bbox num: {bbox_num}")
                
                for i in range(bbox_num):
                    if i >= len(bbox_data):
                        break
                        
                    # Get the bbox array for this detection
                    if len(bbox_data.shape) == 2:
                        bbox = bbox_data[i]
                    else:
                        logger.warning(f"Unexpected bbox shape: {bbox_data.shape}")
                        break
                    
                    # bbox format: [class_id, score, x1, y1, x2, y2]
                    if len(bbox) >= 6:
                        class_id = float(bbox[0])
                        score = float(bbox[1])
                        x1, y1, x2, y2 = float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5])
                        
                        if score >= threshold:
                            bboxes.append([x1, y1, x2, y2])
                            scores.append(score)
                            class_ids.append(int(class_id))
        
        logger.info(f"Detected {len(bboxes)} objects with threshold {threshold}")
        
        return {
            'bboxes': bboxes,
            'scores': scores,
            'class_ids': class_ids,
            'class_names': class_names,
            'num_detections': len(bboxes)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return empty results
        return {
            'bboxes': [],
            'scores': [],
            'class_ids': [],
            'class_names': ['rice_blast'],
            'num_detections': 0
        }


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    # Return file info
    return jsonify({
        'success': True,
        'filename': file.filename,
        'path': filename
    })


@app.route('/api/load_folder', methods=['POST'])
def load_folder():
    """Load images from a folder"""
    data = request.get_json()
    folder_path = data.get('folder_path', '')
    
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return jsonify({'error': 'Invalid folder path'}), 400
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Sort files
    image_files.sort()
    
    return jsonify({
        'success': True,
        'folder_path': folder_path,
        'images': [os.path.basename(f) for f in image_files],
        'total': len(image_files)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run prediction on an image"""
    data = request.get_json()
    image_path = data.get('image_path', '')
    threshold = float(data.get('threshold', 0.5))
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    try:
        # Run prediction
        results = predict_image(image_path, threshold)
        
        # Draw bboxes on image
        result_image = draw_bboxes_on_image(
            image_path, 
            results['bboxes'], 
            results['scores'], 
            results['class_ids'],
            results['class_names'],
            threshold
        )
        
        # Convert image to base64
        buffered = BytesIO()
        result_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_str}",
            'detections': results['num_detections'],
            'results': {
                'bboxes': results['bboxes'],
                'scores': results['scores'],
                'class_ids': results['class_ids'],
                'class_names': results['class_names']
            }
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_image', methods=['GET'])
def get_image():
    """Get an image file"""
    image_path = request.args.get('path', '')
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    # Convert image to base64
    with open(image_path, 'rb') as f:
        img_data = f.read()
        img_str = base64.b64encode(img_data).decode()
    
    return jsonify({
        'success': True,
        'image': f"data:image/jpeg;base64,{img_str}"
    })


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Rice Blast Disease Detection Web App')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU for inference')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    
    args = parser.parse_args()
    
    # Initialize model
    init_model(args.config, args.weights, args.use_gpu)
    
    # Run app
    # Note: Set use_reloader=False to avoid reinitializing the model
    # which can cause issues with global state
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
