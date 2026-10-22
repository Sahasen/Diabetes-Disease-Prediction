# Diabetes Disease Prediction

A machine learning web application that predicts the likelihood of diabetes from diagnostic health measurements, built with Python and Streamlit.

🔗 **Live Demo:** https://diabetes-prediction-sahasen.streamlit.app/

> **Disclaimer:** This project was built for **educational and learning purposes only**. It is not a medical diagnosis tool and must not be used as a substitute for professional medical advice.

---

## Overview

This project trains a Support Vector Machine (SVM) classifier on the Pima Indians Diabetes dataset and serves it through an interactive web interface. Users can enter health parameters and receive an instant prediction along with a probability score. The app also includes small health utilities and data visualizations.

## Features

- **Diabetes Prediction:** enter 8 health parameters and get a prediction with a probability percentage
- **Glucose Level Analysis:** classifies glucose input as normal, elevated, high or critical
- **BMI Calculator:** calculates BMI from height and weight
- **BP, Glucose and Insulin Estimators:** simple rule-based estimators (illustrative only, not clinical formulas)
- **Data Visualizations:** feature distributions split by outcome, with the user's input marked on the chart
- **Model Performance:** displays training and test accuracy
- **Outcome Distribution:** shows the diabetic vs. non-diabetic split in the dataset

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML / Data | scikit-learn, pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud, GitHub |

## Dataset

- **Source:** Pima Indians Diabetes Dataset (originally from the National Institute of Diabetes and Digestive and Kidney Diseases), available on Kaggle
- **Features (8):** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Target:** `Outcome` (1 = diabetic, 0 = non-diabetic)

## Machine Learning Workflow

1. Loaded and explored the dataset
2. Standardized the features using `StandardScaler`
3. Split the data into 80% training and 20% testing sets
4. Trained an SVM classifier (linear kernel) with probability estimates enabled
5. Evaluated the model using accuracy on the train and test sets
6. Deployed the trained model as a Streamlit web app

## Project Structure

```
Diabetes-Disease-Prediction/
├── app.py                              # Streamlit application
├── diabetes.csv                        # Dataset
├── Diabete Disease Predicition.ipynb   # Model development notebook
├── img.jpeg                            # App image
├── requirements.txt                    # Python dependencies
└── README.md
```

## Run Locally

1. **Clone the repository**
```bash
   git clone https://github.com/Sahasen/Diabetes-Disease-Prediction.git
   cd Diabetes-Disease-Prediction/Diabetes-Disease-Prediction
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Run the app**
```bash
   python -m streamlit run app.py
```

## Limitations

- The dataset is small and contains only female patients of Pima Indian heritage, so the model may not generalize to other populations
- The BP, glucose and insulin estimators use simplified formulas for demonstration
- Predictions are not clinically validated

## Author

https://github.com/Sahasen

## Purpose

This project was created for learning purposes to demonstrate an end-to-end machine learning workflow, from data preprocessing and model training to web deployment.
