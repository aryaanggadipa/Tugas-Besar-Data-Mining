
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image
import pickle
from imblearn.over_sampling import SMOTE


pickel_in = open('linear_model.pkl', 'rb')
linearReg = pickle.load(pickel_in)

# Buat scaler dengan parameter yang sesuai dengan data training
scaler = StandardScaler()
# Definisikan mean dan std yang digunakan saat training
# Sesuaikan nilai ini dengan mean dan std dari data training Anda
scaler.mean_ = np.array([
    30.5,  # Age
    0.5,   # Gender (encoded: 0=Male, 1=Female)
    70.0,  # Weight
    1.7,   # Height
    150.0, # Max_BPM
    130.0, # Avg_BPM
    70.0,  # Resting_BPM
    1.0,   # Session_Duration
    300.0, # Calories_Burned
    1.5,   # Workout Type (encoded: 0=Cardio, 1=Strength, 2=Yoga, 3=HIIT)
    20.0,  # Fat_Percentage
    2.0,   # Water_Intake
    3.5,   # Workout_Frequency
    2.0    # Experience Level (encoded: 1=Beginner, 2=Intermediate, 3=Advanced)
])

scaler.scale_ = np.array([
    10.0,  # Age
    0.5,   # Gender
    15.0,  # Weight
    0.1,   # Height
    20.0,  # Max_BPM
    15.0,  # Avg_BPM
    10.0,  # Resting_BPM
    0.5,   # Session_Duration
    100.0, # Calories_Burned
    1.0,   # Workout Type
    5.0,   # Fat_Percentage
    1.0,   # Water_Intake
    1.5,   # Workout_Frequency
    0.8    # Experience Level
])

# Buat scaler khusus untuk BMI
bmi_scaler = StandardScaler()
# Sesuaikan dengan mean dan std BMI dari data training
bmi_scaler.mean_ = np.array([25.0])  # mean BMI dari data training
bmi_scaler.scale_ = np.array([5.0])  # std BMI dari data training

# Buat Fungsi yang dapat mengeluarkan hasil prediksi dari model berdasarkan input dari user
def predict_bmi(features):
    input_data = np.array([list(features.values())])
    # Standardisasi input
    input_scaled = scaler.transform(input_data)
    # Prediksi (hasilnya dalam bentuk scaled)
    pred_scaled = linearReg.predict(input_scaled)
    # Inverse transform hasil prediksi ke nilai BMI asli
    pred_original = bmi_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return pred_original[0][0]

# Buat fungsi yang dapat mengeluarkan metrik evaluasi model (JANGAN DI HAPUS, ISI DIBAWAH INI)
def evaluate_model(X_test, y_test):
  y_pred = linearReg.predict(X_test)
  y_pred_proba = linearReg.predict_proba(X_test)[:, 1]
  mse = mean_squared_error(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  r2_test = r2_score(y_test, y_pred)
  return mse, mae, r2_test, y_pred, y_pred_proba

def main():
  st.title("LINEAR REGRESSION UNTUK MEMPREDIKSI NILAI BMI BERDASARKAN GYM EXERCISE")

  Age = st.text_input("Age")
  Gender = st.selectbox("Gender", ["Male", "Female"])
  Weight = st.text_input("Weight (kg)")
  Height = st.text_input("Height (m)")
  Max_BPM = st.text_input("Max_BPM")
  Avg_BPM = st.text_input("Avg_BPM")
  Resting_BPM = st.text_input("Resting_BPM")
  Session_Duration = st.text_input("Session_Duration (hours)")
  Calories_Burned = st.text_input("Calories_Burned")
  Workout_Type = st.selectbox("Workout Type", ["Cardio", "Strength", "Yoga", "HIIT"])
  Fat_Percentage = st.text_input("Fat_Percentage")
  Water_Intake = st.text_input("Water_Intake (liters)")
  Workout_Frequency = st.text_input("Workout_Frequency (days per week)")
  Experience_Level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])

  result = ""
  proba_result = ""

  if st.button("Predict"):
    # Konversi fitur kategorikal ke numerik =========================================
    gender_encoded = 0 if Gender == "Male" else 1
    workout_type_mapping = {"Cardio": 0, "Strength": 1, "Yoga": 2, "HIIT": 3}
    workout_type_encoded = workout_type_mapping[Workout_Type]
    experience_level_mapping = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    experience_level_encoded = experience_level_mapping[Experience_Level]
    # ================================================================================

    # Gabungkan fitur-fitur ke dalam dictionary
    features = {
        "Age": float(Age),
        "Gender": gender_encoded,
        "Weight (kg)": float(Weight),
        "Height (m)": float(Height),
        "Max_BPM": float(Max_BPM),
        "Avg_BPM": float(Avg_BPM),
        "Resting_BPM": float(Resting_BPM),
        "Session_Duration (hours)": float(Session_Duration),
        "Calories_Burned": float(Calories_Burned),
        "Workout Type": workout_type_encoded,
        "Fat_Percentage": float(Fat_Percentage),
        "Water_Intake (liters)": float(Water_Intake),
        "Workout_Frequency (days per week)": float(Workout_Frequency),
        "Experience Level": experience_level_encoded,
    }
    # ================================================================================

    try:
        # Prediksi BMI
        bmi_result = predict_bmi(features)
        
        # Tampilkan hasil BMI
        st.write("Predicted BMI:", f"{bmi_result:.2f}")
        
        # Kategorisasi BMI
        if bmi_result < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi_result < 25:
            category = "Normal weight"
        elif 25 <= bmi_result < 30:
            category = "Overweight"
        else:
            category = "Obese"
        
        st.write("BMI Category:", category)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam prediksi: {str(e)}")




# __main__
if __name__=='__main__':
    main()
