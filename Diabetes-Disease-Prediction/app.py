# Run the code by this cmd- "python -m streamlit run app.py"

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# load the diabetes dataset
diabetes_df = pd.read_csv('diabetes.csv')

# group the data by outcome to get a sense of the distribution
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# split the data into input and target variables
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# scale the input variables using StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create an SVM model with a linear kernel
model = svm.SVC(kernel='linear', probability=True)

# train the model on the training set
model.fit(X_train, y_test)

# predictions
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

# accuracy
train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)


# ---------------- FUNCTIONS ----------------
def calculate_bmi(weight, height):
    return round(weight / ((height / 100) ** 2), 2) if height > 0 else 0


def estimate_bp(age, bmi, activity_level):
    base_bp = 80 + (bmi * 0.5)
    if age > 40:
        base_bp += 5
    if activity_level == "Low":
        base_bp += 5
    return round(base_bp, 1)


def estimate_glucose(diet, bmi, age, activity_level):
    glucose = 80 + (bmi * 0.8) - (age * 0.1)
    if diet == "High Sugar":
        glucose += 10
    if activity_level == "Low":
        glucose += 5
    return round(glucose, 1)


def estimate_insulin(bmi, diet, age, activity_level):
    insulin = 30 + (bmi * 0.2) - (age * 0.05)
    if diet == "High Sugar":
        insulin += 5
    if activity_level == "Low":
        insulin += 3
    return round(insulin, 1)


# ---------------- APP ----------------
def app():

    img = Image.open(r"img.jpeg")
    img = img.resize((200, 200))
    st.image(img, caption="Diabetes Image", width=200)

    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Go to",
        ["BMI Calculator", "BP Estimator", "Glucose Estimator",
         "Insulin Estimator", "Diabetes Prediction"]
    )

    # BMI
    if page == "BMI Calculator":
        st.title("BMI Calculator")
        height = st.number_input('Height (cm)', 50, 250, 170)
        weight = st.number_input('Weight (kg)', 10, 200, 70)
        bmi = calculate_bmi(weight, height)
        st.write(f'**Your BMI:** {bmi}')

    # BP
    elif page == "BP Estimator":
        st.title("Blood Pressure Estimator")
        age = st.number_input('Age', 10, 100, 30)
        bmi = st.number_input('BMI:', 10.0, 50.0, 22.0)
        activity = st.selectbox('Activity Level', ["High", "Moderate", "Low"])
        bp = estimate_bp(age, bmi, activity)
        st.write(f'**Estimated Blood Pressure:** {bp} mmHg')

    # Glucose
    elif page == "Glucose Estimator":
        st.title("Glucose Estimator")
        bmi = st.number_input('BMI:', 10.0, 50.0, 22.0)
        diet = st.selectbox('Diet Type', ["Balanced", "High Sugar", "Low Carb"])
        age = st.number_input('Age', 10, 100, 30)
        activity = st.selectbox('Activity Level', ["High", "Moderate", "Low"])
        glucose = estimate_glucose(diet, bmi, age, activity)
        st.write(f'**Estimated Glucose Level:** {glucose} mg/dL')

    # Insulin
    elif page == "Insulin Estimator":
        st.title("Insulin Estimator")
        bmi = st.number_input('BMI:', 10.0, 50.0, 22.0)
        diet = st.selectbox('Diet Type', ["Balanced", "High Sugar", "Low Carb"])
        age = st.number_input('Age', 10, 100, 30)
        activity = st.selectbox('Activity Level', ["High", "Moderate", "Low"])
        insulin = estimate_insulin(bmi, diet, age, activity)
        st.write(f'**Estimated Insulin Level:** {insulin}')

    # Diabetes Prediction
    elif page == "Diabetes Prediction":

        st.title("Diabetes Prediction")

        preg = st.slider('Pregnancies', 0, 17, 3)
        glucose = st.slider('Glucose', 0, 400, 117)
        bp = st.slider('Blood Pressure', 0, 122, 72)
        skinthickness = st.slider('Skin Thickness', 0, 99, 23)
        insulin = st.slider('Insulin', 0, 846, 30)
        bmi = st.slider('BMI', 0.0, 67.1, 32.0)
        dpf = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
        age = st.slider('Age', 10, 81, 29)

        # warnings
        if bmi < 18.5:
            st.warning("BMI underweight")
        elif bmi > 40:
            st.warning("BMI high")

        if bp < 60:
            st.warning("BP low")
        elif bp > 140:
            st.warning("BP high")

        if st.button("Predict"):

            input_data = np.array(
                [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
            ).reshape(1, -1)

            st.markdown("### 📊 Glucose Level Analysis")

            if glucose >= 200:
                st.error("CRITICAL HIGH")
            elif glucose >= 126:
                st.error("HIGH")
            elif glucose >= 100:
                st.warning("ELEVATED")
            else:
                st.success("NORMAL")

            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)
            proba = model.predict_proba(scaled_input)[0]

            if prediction[0] == 1:
                st.warning(f"May have diabetes ({proba[1]:.2%})")
            else:
                st.success(f"No diabetes ({proba[0]:.2%})")

        st.header("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Train Accuracy", f"{train_acc:.2%}")
        col2.metric("Test Accuracy", f"{test_acc:.2%}")

    # ---------------- DATA ----------------
    st.header("Distribution by Outcome")

    outcome_counts = diabetes_df["Outcome"].value_counts()
    total = len(diabetes_df)

    col1, col2 = st.columns(2)
    col1.metric("Non-Diabetic", outcome_counts[0], f"{outcome_counts[0]/total*100:.1f}%")
    col2.metric("Diabetic", outcome_counts[1], f"{outcome_counts[1]/total*100:.1f}%")

    # ---------------- FIXED VISUALIZATION ----------------
    if st.checkbox("Show Data Visualizations"):

        st.header("Data Visualizations")

        feature = st.selectbox(
            "Select Feature to Visualize",
            ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin',
             'SkinThickness', 'DiabetesPedigreeFunction', 'Pregnancies']
        )

        # 🔴 FIX: correct session values mapping (NO ERROR NOW)
        mapping = {
            'Glucose': glucose,
            'BMI': bmi,
            'Age': age,
            'BloodPressure': bp,
            'Insulin': insulin,
            'SkinThickness': skinthickness,
            'DiabetesPedigreeFunction': dpf,
            'Pregnancies': preg
        }

        user_value = mapping.get(feature, None)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=diabetes_df, x=feature, hue='Outcome', multiple="stack", ax=ax)

        if user_value is not None:
            ax.axvline(user_value, color='red', linestyle='--')

        st.pyplot(fig)


if __name__ == "__main__":
    app()