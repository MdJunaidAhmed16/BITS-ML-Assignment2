# Multi-Class Classification of Dry Bean Varieties

## 1. Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict the variety of dry beans based on their geometric and morphological features. The project demonstrates a complete end-to-end machine learning workflow including model training, evaluation, deployment using Streamlit, and GitHub version control.

---

## 2. Dataset Description

The dataset used is the **Dry Bean Dataset** from the UCI Machine Learning Repository.

- Total instances: 13,611
- Number of features: 16 numeric attributes
- Target variable: Bean variety (7 classes)
- Problem type: Multi-class classification

Each sample represents a bean characterized by shape, area, perimeter, compactness, and other morphological features extracted from images.

---

## 3. Models Implemented

The following six classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

### Evaluation Metrics Used
- Accuracy
- AUC Score
- Precision (Macro)
- Recall (Macro)
- F1 Score (Macro)
- Matthews Correlation Coefficient (MCC)

---

## 4. Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9207 | 0.9948 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree | 0.9071 | 0.9741 | 0.9227 | 0.9188 | 0.9206 | 0.8876 |
| KNN | 0.9166 | 0.9860 | 0.9318 | 0.9271 | 0.9292 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Random Forest (Ensemble) | 0.9203 | 0.9939 | 0.9349 | 0.9307 | 0.9327 | 0.9036 |
| XGBoost (Ensemble) | 0.9225 | 0.9953 | 0.9366 | 0.9329 | 0.9347 | 0.9063 |

---

## 5. Observations on Model Performance

| ML Model | Observation |
|----------|------------|
| Logistic Regression | Performs strongly due to near-linear separability of classes and well-scaled numeric features. Provides excellent probability estimates. |
| Decision Tree | Captures non-linear relationships but shows slightly lower generalization due to higher variance. |
| KNN | Performs reasonably well but sensitive to feature scaling and dimensionality. Inference is slower compared to other models. |
| Gaussian Naive Bayes | Fast and simple model but limited by the independence assumption between features. |
| Random Forest | Improves stability and performance over a single tree by averaging multiple models and reducing variance. |
| XGBoost | Provides the best overall performance due to gradient boosting, better handling of complex decision boundaries, and strong regularization. |

---

## 6. Streamlit Web Application

The trained models were deployed using Streamlit.  
The app supports:

- Uploading test CSV files
- Selecting different trained models
- Viewing predictions
- Displaying confusion matrix and classification report when true labels are provided
- Downloading a sample test dataset

Live App Link: https://2025aa05431mlassignment.streamlit.app/

---

## 7. Repository Structure
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│ ├── model_building.ipynb
│ ├── *.pkl files
│-- data/
│ └── drybean_test_sample.csv


---

## 8. Execution on BITS Virtual Lab

The notebook and model training were executed on the BITS Virtual Lab environment.  
A screenshot of the execution is included in the final submission PDF as proof.

---

## 9. How to Run Locally

```pip install -r requirements.txt
streamlit run app.py```


Author : Mohammed Junaid Ahmed
BITS-ID : 2025AA05431@wilp.bits-pilani.ac.in

