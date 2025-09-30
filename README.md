# Plant Disease Detection Using Deep Learning

AI-powered plant disease detection system using transfer learning with EfficientNet-B0 to classify diseases in tomato, potato, and pepper plants.

## Overview

This project addresses the critical agricultural challenge of automated plant disease detection using deep learning. Plant diseases cause billions of dollars in crop losses annually, and traditional identification methods rely heavily on expert visual inspection, which is time-intensive and often unavailable in remote areas.

Our solution employs transfer learning with pre-trained convolutional neural networks to classify plant diseases from leaf images. The system compares two state-of-the-art architectures (ResNet50 and EfficientNet-B0) and demonstrates practical applicability for real-world agricultural monitoring.

### Key Achievements
- Developed an automated disease detection system achieving 94.58% test accuracy
- Successfully compared and optimized two CNN architectures for agricultural applications
- Created a comprehensive evaluation framework across 15 disease classes
- Built a production-ready web application with real-time inference capabilities
- Demonstrated practical deployment through Hugging Face Spaces integration

### Problem Significance
- **Agricultural Impact**: Early disease detection enables timely intervention and crop loss prevention
- **Accessibility**: Democratizes plant pathology expertise for farmers in remote areas
- **Scalability**: Provides rapid analysis capabilities for large-scale agricultural operations
- **Cost Reduction**: Minimizes crop losses through automated monitoring and early warning systems

## Performance Summary

- **Test Accuracy**: 94.58%
- **Architecture**: EfficientNet-B0 (Winner)
- **Dataset**: PlantVillage (20,638 images, 15 classes)
- **Training Approach**: Transfer learning with feature extraction

## Model Comparison Results

| Model | Validation Accuracy | Test Accuracy | Training Time | Parameters |
|-------|-------------------|---------------|---------------|------------|
| ResNet50 | 93.25% | - | 344.82s | 30,735 |
| EfficientNet-B0 | 94.64% | 94.58% | 358.51s | 19,215 |

**Winner**: EfficientNet-B0 (+1.39% accuracy advantage)

## Detailed Performance Metrics

- **Weighted Precision**: 0.9459
- **Weighted Recall**: 0.9458  
- **Weighted F1-Score**: 0.9453
- **Macro Precision**: 0.9405
- **Macro Recall**: 0.9406
- **Macro F1-Score**: 0.9400

## Class Performance Analysis

- **Excellent Performance (F1 > 0.95)**: 7 classes
- **Good Performance (F1 > 0.90)**: 7 classes  
- **Need Improvement (F1 < 0.90)**: 1 class
- **Best Class**: Tomato healthy (F1: 0.991)
- **Most Challenging**: Tomato Early blight (F1: 0.806)

## Dataset Information

- **Total Images**: 20,638
- **Disease Classes**: 15
- **Plants Covered**: Tomato (10 classes), Potato (3 classes), Pepper (2 classes)
- **Class Imbalance**: 21.1:1 ratio (largest: 3,208 images, smallest: 152 images)
- **Data Split**: 70% train, 15% validation, 15% test

## Technical Implementation

### Architecture
- **Base Model**: EfficientNet-B0 pre-trained on ImageNet
- **Transfer Learning**: Feature extraction approach
- **Final Layer**: Modified for 15-class classification

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 5
- **Scheduler**: ReduceLROnPlateau

### Data Preprocessing
- **Image Size**: 224×224 pixels
- **Augmentation**: Random flip, rotation, color jitter, affine transforms
- **Normalization**: ImageNet statistics

## Project Structure

```
plant-disease-detection/
├── notebook/
│   └── plant-disease-detection.ipynb    # Complete training notebook
├── models/
│   └── efficientnet_plant_classifier.pth # Trained model
├── backend/
│   └── app.py                           # Hugging Face Spaces backend
├── frontend/
│   └── index.html                       # Web application
└── README.md
```

## Live Demo

- **Backend API**: [Hugging Face Spaces](https://huggingface.co/spaces/mrtechnomix/plant-diseas)
- **Web Application**: Modern UI with real-time disease detection

## Usage

### Training
```python
# Load and train the model
python train_model.py
```

### Inference
```python
# Use the trained model for prediction
from plant_disease_detector import predict_disease
result = predict_disease(image_path)
```

### Web Application
Open `frontend/index.html` in a browser or deploy to any web server.

## Requirements

```
torch>=2.0.1
torchvision>=0.15.2
gradio>=4.7.1
Pillow>=10.0.1
numpy>=1.24.3
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

## Installation

```bash
git clone <repository-url>
cd plant-disease-detection
pip install -r requirements.txt
```

## Results Highlights

- Achieved 94.58% test accuracy on plant disease classification
- EfficientNet-B0 outperformed ResNet50 with fewer parameters
- Strong performance across 14 out of 15 disease classes
- Ready for deployment in agricultural applications

## Applications

- Early disease detection for crop management
- Agricultural extension services
- Precision farming initiatives
- Educational tools for plant pathology

## Future Improvements

- Address class imbalance with advanced sampling techniques
- Expand dataset with more diverse field conditions
- Implement ensemble methods for improved accuracy
- Add confidence thresholding for uncertain predictions
