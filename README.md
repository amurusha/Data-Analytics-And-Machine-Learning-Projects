# Data-Analytics-And-Machine-Learning-Projects
## Overview

This repository contains projects focused on **Data Analytics and Machine Learning**. The primary objective is to analyze datasets, apply machine learning models, and draw meaningful insights.
[Data Analytics and Machine Learning.pdf](https://github.com/user-attachments/files/18559740/Data.Analytics.and.Machine.Learning.pdf)
---
## Project 1: Extra Sensory Data Analysis

### Dataset
The dataset consists of sensor measurements collected from personal smartphones and smartwatches of 60 users.

### Key Objectives:
- Improve the test set.
- Introduce a validation set.
- Use data splitting techniques to increase training data.
- Experiment with multiple models and evaluate results.

### Techniques:
- **Logistic Regression**: Analyzed performance with varying regularization parameters.
- **Data Split Strategy**: 80-20 split for training and validation.
- **Performance Metrics**: 
  - Test Accuracy
  - Balanced Accuracy
  - Variance

### Tools:
- Python Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

---

## Project 2: Planet Four - CNN Implementation

### Dataset
Images of the Martian surface from the southern polar region, aiming to detect features such as fans and blotches.

### Key Objectives:
- Train Convolutional Neural Networks (CNNs) to classify Martian surface features.
- Experiment with ResNet architectures (ResNet50, ResNet101, ResNet152).
- Analyze overfitting and underfitting patterns.

### Techniques:
- **Data Augmentation**: Random mirroring for better generalization.
- **Model Comparison**: 
  - ResNet50 with SGD and Adam optimizers.
  - ResNet101 and ResNet152.
- **Loss Functions**: Binary Cross-Entropy Loss.
- **Validation**: ROC-AUC, classification reports, and accuracy metrics.

### Tools:
- Python Libraries: `torch`, `torchvision`, `pandas`, `matplotlib`.
- Environment: Google Colab with GPU acceleration.

---

## Installation and Requirements

### Prerequisites:
- Python 3.7+
- Libraries:
  ```
  pandas, numpy, scikit-learn, matplotlib, torch, torchvision
  ```

### Setup:
1. Clone this repository:
   ```
   git clone https://github.com/username/repo-name.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## How to Run

1. Navigate to the respective project folder:
   - `extrasensory-data/`
   - `planet-four-cnn/`
2. Run the provided Jupyter Notebooks or Python scripts for each task.

---

## Results and Insights

### Extra Sensory Data:
- Test Accuracy: Improved from baseline to a balanced accuracy mean of **81.85%**.
- Logistic Regression was found effective with proper data preprocessing.

### Planet Four - CNN:
- Overfitting observed across ResNet50, ResNet101, and ResNet152.
- ResNet50 with SGD showed better validation accuracy, though the model remains suboptimal for this dataset.

