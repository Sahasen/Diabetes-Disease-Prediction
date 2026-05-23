# Diabetes-Disease-Prediction-Model


This project is a machine learning-based web application that predicts whether a person is diabetic or not using clinical data such as glucose level, BMI, insulin, and more.

## 📁 Files and Description

### 1. `Diabete Disease Predicition.ipynb`
- Jupyter Notebook used for:
  - Loading and preprocessing the dataset
  - Training ML models: Logistic Regression, Random Forest, and SVM
  - Evaluating model accuracy
  - Plotting model comparison graph.
  
### 2. `app.py`
- Streamlit web application file
- Allows users to:
  - Estimate BMI, blood pressure, glucose, and insulin
  - Enter clinical values for diabetes prediction
  - Visualize input against dataset distributions
  - Get model predictions along with warnings based on glucose levels

### 3. `diabetes.csv`
- Dataset containing health-related metrics for diabetes prediction.
- Sourced from the Pima Indians Diabetes Dataset.


### 4.`img.jpeg`
- Project-related image (e.g., output graph or visual content for report). 



 # 🚀 Features

- Predicts whether a patient has diabetes based on inputs like glucose, insulin, BMI, etc.
- Real-time prediction using trained machine learning models.
- Glucose level classification and warnings.
- Body fat calculator component (in JS).
- Visual comparison of model accuracy.
- Streamlit-based interactive UI.



## ⚙️ How to Run the Project

### 📌 Requirements
- Python 3.x
- Install required libraries:
  ```bash
  pip install numpy pandas, scikit-learn, streamlit, matplotlib ,seaborn

  To run the code(app.py)------python -m streamlit run app.py

