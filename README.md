# MNIST Digit Classification: KNN vs SVM vs Neural Network

A comparison of three classical and deep-learning approaches on handwritten digit recognition, built for the COE292 machine learning course at KFUPM. The notebook walks through preprocessing, model tuning, evaluation and a side-by-side comparison on the same 80/20 split.

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## Results

All models are evaluated on the same held-out test set of 14,000 images (80/20 split of 70,000 samples).

| Model | Configuration | Test accuracy |
|---|---|---|
| **SVM** | RBF kernel, `C=1.0`, `gamma='scale'` on raw pixels | **97.8%** (5-fold CV mean: 97.7%) |
| **Neural network** | Dense 64-32-16 + softmax, Adam (lr 0.001), early stopping | 96.95% (5-fold CV mean: 96.2%) |
| **KNN** | K=3, Manhattan distance, standardised features | 95.9% |

Other experiments in the notebook:

| Experiment | Result |
|---|---|
| KNN with Euclidean distance (K=3) | 94.7% |
| SVM on 50 PCA components, linear / polynomial / RBF kernels (training subset) | 90.3% / 90.9% / 93.9% |
| SVM, RBF kernel on standardised full data | 96.6% |

**Takeaways:** the RBF-SVM was the strongest model here, and the neural network was close behind. Manhattan distance beat Euclidean for KNN on this pixel data. In this notebook the SVM scored higher on raw pixel intensities than on standardised features (97.8% vs 96.6%), a reminder to validate preprocessing per model rather than apply it by default.

## What the notebook covers

1. **Data loading and preprocessing**: 70,000 samples, 785 columns (label plus 784 pixel values), 80/20 train/test split, `StandardScaler`, visual comparison of original vs normalised images.
2. **KNN**: error rate vs K (1–20), Euclidean vs Manhattan distance, confusion matrices, classification reports and cross-validation.
3. **SVM**: PCA to 50 components, kernel comparison (linear, polynomial, RBF), final model with 5-fold cross-validation.
4. **Neural network**: a small fully connected Keras network with early stopping, evaluated on the test set and with stratified 5-fold cross-validation.
5. **Comparison**: a bar chart of KNN, SVM and neural-network accuracy.

## Running it

The dataset (`dataset_7.csv`) was provided by the course and is **not included** in this repository. Place it next to the notebook, then:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
jupyter notebook "COE292 Project Final.ipynb"
```

## Tech stack

Python, NumPy, pandas, scikit-learn (KNN, SVM, PCA, metrics, cross-validation), TensorFlow/Keras, Matplotlib, Seaborn.

## Repository structure

```
.
├── COE292 Project Final.ipynb   # full analysis, from preprocessing to model comparison
└── README.md
```
