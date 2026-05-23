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
model.fit(X_train, y_train)
# make predictions on the training and testing sets
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)
# calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)


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

def app():
        
        
        img = Image.open(r"img.jpeg")
        img = img.resize((200,200))
        st.image(img,caption="Diabetes Image",width=200)
        
        st.sidebar.title('Navigation')
        page = st.sidebar.radio("Go to", ["BMI Calculator", "BP Estimator", "Glucose Estimator", "Insulin Estimator", "Diabetes Prediction"])
        
        if page == "BMI Calculator":
            st.title("BMI Calculator")
            height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170)
            weight = st.number_input('Weight (kg)', min_value=10, max_value=200, value=70)
            bmi = calculate_bmi(weight, height)
            st.write(f'**Your BMI:** {bmi}')
        
        elif page == "BP Estimator":
            st.title("Blood Pressure Estimator")
            age = st.number_input('Age', min_value=10, max_value=100, value=30)
            bmi = st.number_input('BMI:', min_value=10.0, max_value=50.0, value=22.0)
            activity = st.selectbox('Activity Level', ["High", "Moderate", "Low"])
            bp = estimate_bp(age, bmi, activity)
            st.write(f'**Estimated Blood Pressure:** {bp} mmHg')
        
        elif page == "Glucose Estimator":
            st.title("Glucose Estimator")
            bmi = st.number_input('BMI:', min_value=10.0, max_value=50.0, value=22.0)
            diet = st.selectbox('Diet Type', ["Balanced", "High Sugar", "Low Carb"])
            age = st.number_input('Age', min_value=10, max_value=100, value=30)
            activity = st.selectbox('Activity Level', ["High", "Moderate", "Low"])
            glucose = estimate_glucose(diet, bmi, age, activity)
            st.write(f'**Estimated Glucose Level:** {glucose} mg/dL')
        
        elif page == "Insulin Estimator":
            st.title("Insulin Estimator")
            bmi = st.number_input('BMI:', min_value=10.0, max_value=50.0, value=22.0)
            diet = st.selectbox('Diet Type', ["Balanced", "High Sugar", "Low Carb"])
            age = st.number_input('Age', min_value=10, max_value=100, value=30)
            activity = st.selectbox('Activity Level', ["High", "Moderate", "Low"])
            insulin = estimate_insulin(bmi, diet, age, activity)
            st.write(f'**Estimated Insulin Level:** {insulin}')
        
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
            
            # Basic warnings for extreme values
            if bmi < 18.5:
                st.warning("⚠️ BMI is underweight. Please verify the input.")
            elif bmi > 40:
                st.warning("⚠️ BMI indicates severe obesity. Please verify the input.")
            if bp < 60:
                st.warning("⚠️ Blood pressure is unusually low. Please verify the input.")
            elif bp > 140:
                st.warning("⚠️ Blood pressure is unusually high. Please verify the input.")
            
            if st.button("Predict"):
                input_data = np.array([preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]).reshape(1, -1)
                
                # Detailed Glucose Analysis Section
                st.markdown("### 📊 Glucose Level Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Glucose Level:**")
                    st.markdown(f"### {glucose} mg/dL")
                    
                    if glucose >= 200:
                        st.error("⚠️ CRITICAL HIGH LEVEL")
                    elif glucose >= 126:
                        st.error("⚠️ HIGH LEVEL")
                    elif glucose >= 100:
                        st.warning("⚠️ ELEVATED LEVEL")
                    elif glucose >= 70:
                        st.success("✅ NORMAL LEVEL")
                    else:
                        st.warning("⚠️ LOW LEVEL")
                
                with col2:
                    st.markdown("**Medical Guidelines:**")
                    st.markdown("""
                    - Normal: 70-99 mg/dL
                    - Pre-diabetes: 100-125 mg/dL
                    - Diabetes: ≥126 mg/dL
                    - Critical: ≥200 mg/dL
                    """)
                
                st.markdown("**Detailed Analysis:**")
                if glucose >= 200:
                    st.error("🚨 CRITICAL HIGH: Glucose level ≥200 mg/dL indicates definite diabetes")
                    st.write("- This level is diagnostic for diabetes regardless of other factors")
                    st.write("- Immediate medical consultation is strongly recommended")
                    st.write("- Symptoms may include excessive thirst, frequent urination, and fatigue")
                elif glucose >= 126:
                    st.error("⚠️ HIGH: Glucose level ≥126 mg/dL indicates probable diabetes")
                    st.write("- This level suggests diabetes if confirmed by repeat testing")
                    st.write("- Medical consultation is recommended")
                    st.write("- Lifestyle changes and monitoring are important")
                elif glucose >= 100:
                    st.warning("⚠️ ELEVATED: Glucose level 100-125 mg/dL indicates pre-diabetes")
                    st.write("- This range suggests increased risk of developing diabetes")
                    st.write("- Lifestyle modifications can help prevent progression to diabetes")
                    st.write("- Regular monitoring is recommended")
                elif glucose >= 70:
                    st.success("✅ NORMAL: Glucose level 70-99 mg/dL is in healthy range")
                    st.write("- This is considered a healthy fasting glucose level")
                    st.write("- Continue maintaining a healthy lifestyle")
                else:
                    st.warning("⚠️ LOW: Glucose level below 70 mg/dL indicates hypoglycemia")
                    st.write("- This level is unusually low and may require immediate attention")
                    st.write("- Common symptoms include shakiness, sweating, and confusion")
                
                # Model Prediction Section
                st.markdown("### 🔄 Model Prediction")
                if glucose >= 200:
                    st.error('⚠️ This person **has diabetes** (Based on glucose level ≥200 mg/dL)')
                    st.write('This is a definitive diagnosis based on medical standards.')
                    st.write('Recommendation: Immediate medical consultation is advised.')
                elif glucose >= 126:
                    scaled_input = scaler.transform(input_data)
                    prediction = model.predict(scaled_input)
                    prediction_proba = model.predict_proba(scaled_input)[0]
                    
                    st.error('⚠️ This person likely **has diabetes** (Based on glucose level ≥126 mg/dL)')
                    st.write(f'Model Probability of having diabetes: {prediction_proba[1]:.2%}')
                    st.write('Recommendation: Medical consultation is advised.')
                else:
                    scaled_input = scaler.transform(input_data)
                    prediction = model.predict(scaled_input)
                    prediction_proba = model.predict_proba(scaled_input)[0]
                    
                    if prediction[0] == 1:
                        st.warning('⚠️ This person **may have diabetes**')
                        st.write(f'Probability of having diabetes: {prediction_proba[1]:.2%}')
                        st.write('Recommendation: Further medical testing is advised.')
                    else:
                        st.success('✅ This person likely **does not have diabetes**')
                        st.write(f'Probability of not having diabetes: {prediction_proba[0]:.2%}')

            st.header('Model Performance')
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Training Accuracy", f"{train_acc:.2%}")
            with col2:
                st.metric("Testing Accuracy", f"{test_acc:.2%}")
                    
        # Distribution by Outcome with explanation
        st.header('Distribution by Outcome')
        outcome_counts = diabetes_df['Outcome'].value_counts()
        total_cases = len(diabetes_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Non-Diabetic Cases (0)", 
                     f"{outcome_counts[0]}", 
                     f"{(outcome_counts[0]/total_cases)*100:.1f}%")
        with col2:
            st.metric("Diabetic Cases (1)", 
                     f"{outcome_counts[1]}", 
                     f"{(outcome_counts[1]/total_cases)*100:.1f}%")
        
        
        
        # Feature Statistics by Outcome
        st.header('Average Values by Diabetes Outcome')
        
        # Calculate mean values for each feature grouped by outcome
        mean_by_outcome = diabetes_df.groupby('Outcome').mean().round(2)
        
        # Create a more readable format
        comparison_df = pd.DataFrame({
            'Feature': mean_by_outcome.index.map({0: 'Non-Diabetic', 1: 'Diabetic'}),
            'Pregnancies': mean_by_outcome['Pregnancies'],
            'Glucose': mean_by_outcome['Glucose'],
            'Blood Pressure': mean_by_outcome['BloodPressure'],
            'Skin Thickness': mean_by_outcome['SkinThickness'],
            'Insulin': mean_by_outcome['Insulin'],
            'BMI': mean_by_outcome['BMI'],
            'Diabetes Pedigree': mean_by_outcome['DiabetesPedigreeFunction'],
            'Age': mean_by_outcome['Age']
        }).set_index('Feature')
        
        st.write(comparison_df)
        
        st.markdown("""
        **Understanding the Averages:**
        - Shows average values for each feature in diabetic vs non-diabetic cases
        - Helps identify typical differences between outcomes
        - Useful for understanding feature patterns in each group
        """)
        
        # Dataset Summary with explanations
        st.header('Dataset Summary Statistics')
        summary_stats = diabetes_df.describe().round(2)
        st.write(summary_stats)
        
        st.markdown("""
        **Understanding the Summary Statistics:**
        - Count: Number of measurements for each feature
        - Mean: Average value
        - Std: Standard deviation (spread of values)
        - Min/Max: Range of values
        - 25%, 50%, 75%: Quartile values showing data distribution
        """)
        
        if st.checkbox("Show Data Visualizations"):
            st.header("Data Visualizations")
            
            # Distribution plots with user input
            feature = st.selectbox("Select Feature to Visualize", 
                                ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'SkinThickness', 'DiabetesPedigreeFunction', 'Pregnancies'])
            
            # Get current user input value based on selected feature
            user_value = None
            if feature == 'Glucose':
                user_value = glucose
            elif feature == 'BMI':
                user_value = bmi
            elif feature == 'Age':
                user_value = age
            elif feature == 'BloodPressure':
                user_value = bp
            elif feature == 'Insulin':
                user_value = insulin
            elif feature == 'SkinThickness':
                user_value = skinthickness
            elif feature == 'DiabetesPedigreeFunction':
                user_value = dpf
            elif feature == 'Pregnancies':
                user_value = preg

            # Create distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=diabetes_df, x=feature, hue='Outcome', multiple="stack")
            
            # Add vertical line for user input if available
            if user_value is not None:
                plt.axvline(x=user_value, color='red', linestyle='--', label='Your Input')
                plt.legend(['Your Input', 'No Diabetes', 'Has Diabetes'])
            
            plt.title(f"Distribution of {feature} in Dataset vs Your Input")
            st.pyplot(fig)
            
            # Add explanation of the visualization
            st.markdown("""
            **Understanding the Visualization:**
            - The stacked histogram shows the distribution of values in our dataset
            - Blue represents people without diabetes
            - Orange represents people with diabetes
            - The red dashed line shows your input value
            - This helps you see how your values compare to the dataset distribution
            """)

if __name__ == '__main__':
    app()
