# 🧠 Brain Tumor Segmentation using U-Net (ResNet34 + scSE Attention)

## 📌 Overview
This project implements a deep learning pipeline for brain tumor segmentation from CT images using an enhanced U-Net architecture. The model improves a baseline segmentation approach by integrating a pretrained ResNet34 encoder, attention mechanisms, hybrid loss functions, and advanced training strategies.

The final model achieved a Dice Score of 0.5669, showing strong performance given the challenges of small tumor regions and noisy CT data.

---

## 🎯 Objectives
- Build an end-to-end segmentation model for brain tumor detection  
- Improve feature extraction using pretrained encoders  
- Enhance segmentation accuracy using attention mechanisms  
- Handle class imbalance effectively  
- Evaluate performance using Dice score  

---

## 🧠 Model Architecture
- Base Model: U-Net for image segmentation  
- Encoder: ResNet34 pretrained on ImageNet  
- Decoder: Enhanced with scSE (spatial and channel squeeze-and-excitation) attention  
- Output: Single-channel probability mask using sigmoid activation  

---

## ⚙️ Key Features
- Balanced sampling to address class imbalance  
- Data augmentation (flip, rotation, brightness/contrast)  
- Hybrid loss function (Focal BCE + Weighted BCE + Dice Loss)  
- AdamW optimizer with learning rate scheduling  
- Early stopping to prevent overfitting  
- CRF-based post-processing for refined segmentation  

---

## 📊 Dataset
- ADDA Brain CT dataset  
- 2D grayscale CT images (converted to 3-channel)  
- Binary tumor masks  
- Dataset is highly imbalanced  

Note: Dataset is not included due to size constraints.

---

## 📈 Results
- Best Dice Score: 0.5669  

### Observations
- Smooth training convergence  
- Improved tumor localization with attention  
- Better boundary detection  
- Reduced false positives  

---

## 🧪 Inference Pipeline
- Model prediction  
- Thresholding  
- CRF-based refinement  

---

## ▶️ How to Run

### Install dependencies
``` bash 
pip install -r requirements.txt
```

### Train the model
``` bash
python main.py
```

### Run inference
``` bash
python inference.py
```

---

## 🛠️ Tech Stack
- Python  
- PyTorch  
- Albumentations  
- OpenCV  
- NumPy  

---

## 🚀 Key Contributions
- Implemented U-Net with ResNet34 encoder  
- Integrated scSE attention for improved segmentation  
- Designed hybrid loss function for better learning  
- Built a complete pipeline from preprocessing to inference  

---

## Project Report
Detailed explanation is included in the repository.

---


## Academic Use & References
This project was developed as part of a **graduate-level academic coursework** in image processing and deep learning.

The implementation is inspired by:
- Standard U-Net architecture for biomedical segmentation  
- ResNet-based encoder designs  
- Attention mechanisms such as scSE blocks  
- Various research papers and publicly available online resources  

This work is intended for **educational and research purposes only**.

---

## License
This project is for academic use only.  
It may reference publicly available research ideas and learning resources.

You are free to:
- Use it for learning and research  
- Modify for academic purposes  

