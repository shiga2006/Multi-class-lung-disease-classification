Few-Shot Lung Disease Classification via Prototypical Networks

This repository provides a PyTorch implementation of a **Prototypical Network** with a Simple CNN encoder for few-shot multi-class classification of chest X-ray images. The model is designed to distinguish between five classes: **Normal, Bacterial Pneumonia, Viral Pneumonia, COVID-19, and Tuberculosis**. The approach is suitable for scenarios with limited labeled data per class, as commonly found in medical imaging.

Overview

This project implements a few-shot learning approach for medical image classification using Prototypical Networks. The workflow involves episodic sampling, a simple CNN encoder for feature extraction, computation of class prototypes, and classification of query images based on their proximity to these prototypes.


Environment

- **OS:** Windows 10/11 or Ubuntu 20.04+
- **Processor:** Intel Core i7-1185G7 or better
- **RAM:** 32 GB
- **Python:** 3.8+
- **PyTorch:** 1.9+
- **CUDA:** (optional, for GPU acceleration)
- **Other Libraries:** torchvision, numpy, pandas, matplotlib, seaborn, scikit-learn

Dataset

- **Input:** Chest X-ray images organized into 5 classes:
    - Normal
    - Bacterial Pneumonia
    - Viral Pneumonia
    - COVID-19
    - Tuberculosis
- **Structure:**  
    ```
    Lung Disease Dataset/
        train/
            Normal/
            Bacterial Pneumonia/
            Viral Pneumonia/
            COVID/
            Tuberculosis/
        val/
            Normal/
            ...
- **Size:** 6,000 training and 2,000 validation images (customizable)

Methodology

Episodic Training

- Each episode samples `N_WAY` classes, with `N_SUPPORT` support images and `N_QUERY` query images per class.
- Prototypical Networks compute class prototypes from support embeddings and classify queries based on nearest prototype (Euclidean distance).

Model Architecture

- **Simple CNN Encoder:** 3 Conv2D layers + ReLU + MaxPool, AdaptiveAvgPool2d, Flatten, Linear (see [model diagram](#) and `encoder.py`)
- **Prototypical Network:** Computes prototypes and classifies based on embedding distances.

Training

- Optimizer: Adam, LR=1e-3
- Loss: CrossEntropyLoss
- Epochs: 20 (default)
- Best model checkpoint is saved automatically.

![Uploading image.png…]()

Usage

1. Clone the Repository

```bash
git clone https://github.com/shiga2006/Multi-class-lung-disease-classification
```

2. Prepare the Dataset

Organize your chest X-ray images into the folder structure shown above.

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Run Training

Edit the `DATA_DIR` in the main script if needed, then:

```bash
python main.py
```

5. Evaluation & Results

- The best model is saved as `best_protonet_xray_long[5class].pth`
- Training/validation loss curves and confusion matrices are saved in the working directory.
![image](https://github.com/user-attachments/assets/b3028cb3-f7ae-4e26-81cb-b9c9bd6eb7b3)


Results
Overall accuracy - 81.6%
![image](https://github.com/user-attachments/assets/a2a684c8-9152-48e3-b63d-3b9f6ef8d7c8)
![image](https://github.com/user-attachments/assets/5dc6b90b-415b-4a33-b101-ab87612782b4)





License

This project is for academic and research purposes only.
