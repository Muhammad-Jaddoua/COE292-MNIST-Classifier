# COE292-MNIST-Classifier
Repo for the machine learning project for COE292 which solves MNIST classification problem using KNN, SVM, and NN models.
# 🧠 KNN & SVM Image Classification

## 📌 Overview
This project applies **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** models to a dataset of handwritten digit images (`dataset_7.csv`, 28×28 pixels). The dataset contains **70,000 samples** with pixel intensities (0–255) and labels (digits 0–9).

The goal is to preprocess the data, train classifiers, compare KNN with different distance metrics, and explore SVM with dimensionality reduction.

---

## ⚙️ Features
- Data preprocessing and normalization (`StandardScaler`)  
- Visualization of original vs. normalized images  
- **KNN Classifier**  
  - Error rate vs. K visualization  
  - Euclidean and Manhattan distance comparisons  
  - Confusion matrices, classification reports, cross-validation  
- **SVM Classifier**  
  - Dimensionality reduction with PCA  
  - Exploration of hard vs. soft margin SVMs  
  - Visualization of support vectors  

---

## 🛠️ Technologies Used
- **Python 3.x**  
- **Libraries**:  
  - `numpy`, `pandas`  
  - `matplotlib`, `seaborn`  
  - `scikit-learn` (KNN, SVM, PCA, metrics, cross-validation)  

---

## 🚀 Project Workflow
### 1. Load Dataset  
- `dataset_7.csv` with 70,000 entries, 785 columns (1 label + 784 pixel features).  

### 2. Preprocessing  
- Split into training (80%) and testing (20%).  
- Normalize features with `StandardScaler`.  
- Visualize normalization effect on sample images.  

### 3. KNN Classifier  
- Trained with varying **K = 1–20**.  
- Error rate plotted against K.  
- Final models tested with **Euclidean** and **Manhattan** distances.  
- Results:  
  - **Euclidean (K=3)** → Accuracy ≈ **94.7%**  
  - **Manhattan (K=3)** → Accuracy ≈ **95.9%**  

### 4. SVM Classifier  
- PCA reduced features to 50 principal components.  
- Hard and soft margin SVMs tested on sample subsets.  
- Support vectors analyzed and compared.  

---

## 📊 Results
- **KNN (Manhattan, K=3)** achieved the best performance: **95.9% accuracy** with strong precision/recall across all digit classes.  
- **SVM with PCA** showed effective dimensionality reduction, allowing margin experiments with manageable complexity.  

---

## 📂 Repository Structure
