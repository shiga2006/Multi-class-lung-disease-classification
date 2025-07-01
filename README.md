Few-Shot Lung Disease Classification via Prototypical Networks

This repository provides a PyTorch implementation of a **Prototypical Network** with a Simple CNN encoder for few-shot multi-class classification of chest X-ray images. The model is designed to distinguish between five classes: **Normal, Bacterial Pneumonia, Viral Pneumonia, COVID-19, and Tuberculosis**. The approach is suitable for scenarios with limited labeled data per class, as commonly found in medical imaging.

Overview

This project implements a few-shot learning approach for medical image classification using Prototypical Networks. The workflow involves episodic sampling, a simple CNN encoder for feature extraction, computation of class prototypes, and classification of query images based on their proximity to these prototypes.


Environment
The experimentation of this work was conducted in a system with the configuration of 13th Gen Intel® Core™ i5-13500 processor and running a 64-bit Windows operating system on an x64-based architecture. The model was trained using PyTorch, a machine learning framework. The training session comprising 500 episodes for each epoch, twenty epochs were used in the session. Each epoch required approximately 8 to 10 minutes to complete, highlighting the computational demands of episodic training in few-shot learning.

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
  
![Screenshot 2025-06-19 111651](https://github.com/user-attachments/assets/d3157147-4c3d-431d-a8ae-a70a1890287b)


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



Blog - https://meta-learning.hashnode.dev/cracking-the-code-of-rare-cases-meta-learning-in-medical-imaging

License

This project is for academic and research purposes only.
