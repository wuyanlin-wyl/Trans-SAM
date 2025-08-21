# Trans-SAM: Transfer Segment Anything Model to Medical Image Segmentation with Parameter-Efficient Fine-Tuning

Official PyTorch implementation of **"Trans-SAM: Transfer Segment Anything Model to Medical Image Segmentation with Parameter-Efficient Fine-Tuning"**.

## 📖 Abstract

**Trans-SAM** utilizes Parameter-Efficient Fine-Tuning (PEFT) to transfer the Segment Anything Model (SAM) to medical image segmentation tasks. Our method introduces two key innovations:

- **Intuitive Perceptual Fine-tuning (IPF) adapter**: Directly integrates input image features into each encoder layer
- **Multi-scale Domain Transfer (MDT) adapter**: Uses convolution-based mechanisms to infuse inductive biases into SAM

## 🏆 Key Features

- ✅ **High Performance**: Achieves superior results compared to state-of-the-art PEFT methods
- ✅ **Parameter Efficient**: Only requires training a small portion of parameters while maintaining excellent performance  
- ✅ **Multi-domain Support**: Validated on 6 medical datasets across different organs and modalities
- ✅ **Automatic Segmentation**: Performs semantic segmentation without requiring prompts

## 🛠️ Installation

### Requirements

```
Python = 3.10
PyTorch = 2.6.0
```

### Dependencies List

Create a `requirements.txt` file with:

```
numpy>=2.2.6
opencv-python>=4.12.0.88
scikit-learn>=1.7.1
torch>=2.6.0
torchvision>=0.21.0
```

## 📁 Project Structure

```
Trans-SAM/
├── SAM/                          # SAM model implementation
├── model_utils/                  # Utility modules
│   ├── cfg.py                   # Configuration file
│   ├── dataset_split.py         # Dataset handling
│   ├── evalution_segmentation.py # Evaluation metrics
│   └── class_dict.csv            # Label mapping table
├── dataset/                      # Dataset directory
│   └── BUSI/                    # Example dataset
├── weight/                       # Model checkpoints directory
├── train_SAM.py                 # Main training script
├── test_SAM.py                  # Testing script  
├── predict_SAM.py               # Prediction script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 📊 Supported Datasets

| Dataset | Description | Images | Modality | Download Link |
|---------|-------------|--------|----------|---------------|
| **LiTS** | Liver Tumor Segmentation Challenge | 131 | CT | [Link](https://competitions.codalab.org/competitions/17094#participate) |
| **ISIC** | Skin Lesion Segmentation | 2,594 | Dermoscopy | [Link](https://challenge2018.isic-archive.com/) |
| **Kvasir** | Polyp Segmentation | 1,000 | Endoscopy | [Link](https://datasets.simula.no/kvasir-seg/) |
| **BUSI** | Breast Ultrasound Segmentation | 780 | Ultrasound | [Link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) |
| **CXML** | Chest X-ray Segmentation | 138 | X-ray | [Link](http://archive.nlm.nih.gov/) |
| **FML** | Finding and Measuring Lungs | - | CT | [Link](https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data) |

### Dataset Preparation

1. **Download datasets** from the links above
2. **Organize your data** in the following structure:

```
dataset/
├── DATASET_NAME/
│   ├── images/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── masks/
│       ├── image001.png
│       ├── image002.png
│       └── ...
```

3. **Update configuration** in `model_utils/cfg.py`:

```python
# Dataset paths
TRAIN_ROOT = "./dataset/YOUR_DATASET/images"
TRAIN_LABEL = "./dataset/YOUR_DATASET/masks"

# Training parameters
BATCH_SIZE = 32
EPOCH_NUMBER = 200
lr = 0.0001
image_size = 256
```

## 🚀 Quick Start

### 1. Training

```bash
# Train on default dataset (configured in cfg.py)
python train_SAM.py

# Monitor training progress
# Check the console output for loss and metrics
# Model checkpoints will be saved in ./weight/ directory
```

### 2. Testing

```bash
# Test the trained model
python test_SAM.py

# Results will be displayed in console
```

### 3. Prediction

```bash
python predict_SAM.py
```



## 📖 Citation

If you find this work helpful in your research, please consider citing:

```bibtex
@article{wu2025trans,
  title={Trans-sam: Transfer segment anything model to medical image segmentation with parameter-efficient fine-tuning},
  author={Wu, Yanlin and Wang, Zhihong and Yang, Xiongfeng and Kang, Hong and He, Along and Li, Tao},
  journal={Knowledge-Based Systems},
  volume={310},
  pages={112909},
  year={2025},
  publisher={Elsevier}
}
```


## Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [Awesome-Parameter-Efficient-Transfer-Learning Public](https://github.com/facebookresearch/segment-anything)
- PyTorch and open-source deep learning community
