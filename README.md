# 🥗 Obesity Data Analysis

This project focuses on the analysis of a dataset related to **nutritional habits** and **physical condition** of individuals, aiming to extract insights regarding **obesity**.  
The dataset contains over **2,000 records** with variables such as age, gender, eating habits, physical activity, and lifestyle patterns.

The project applies **data preprocessing, visualization, clustering, classification, and regression** techniques using **Python, Scikit-learn, and Keras**.

---

## 📂 Project Structure

data/       
 └── obesity_dataset.csv    
preprocessing.py    
clustering.py    
classification_regression.py   
README.md

- preprocessing.py → Cleans and prepares the dataset (outlier removal, normalization, one-hot encoding, PCA, visualizations).  
- clustering.py → Applies clustering techniques (KMeans, DBSCAN) and visualizes results.  
- classification_regression.py → Performs classification (Random Forest, SVM) and regression (BMI prediction with Neural Networks).  

---

## ▶️ Usage

Run the scripts **in order**:

1. Preprocessing
   python preprocessing.py

2. Clustering
   python clustering.py

3. Classification & Regression
   python classification_regression.py

---

## 📊 Features

- Data Preprocessing
  - Cleaning & handling outliers  
  - Normalization with StandardScaler  
  - One-hot encoding of categorical variables  
  - Correlation heatmaps & PCA for dimensionality reduction  

- Clustering
  - KMeans with elbow method & silhouette score  
  - DBSCAN for density-based clustering & outlier detection  

- Classification
  - Predicting obesity categories with Random Forest and SVM  
  - Evaluation with classification reports & confusion matrices  

- Regression
  - Predicting BMI using Neural Networks (Keras)  
  - Custom Feedforward model vs. Transfer Learning model  
  - Evaluation with MAE, RMSE, MAPE  

---

## 📈 Results (Summary)

- Clustering:  
  - KMeans with 6 clusters → silhouette ≈ 0.35  
  - DBSCAN detected outliers but was more sensitive to parameters  

- Classification:  
  - Random Forest & SVM performed with solid accuracy on predicting obesity categories  

- Regression:  
  - Custom neural network outperformed transfer learning model in BMI prediction  

---

## 🛠️ Tech Stack

- Python 3.x
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow / Keras