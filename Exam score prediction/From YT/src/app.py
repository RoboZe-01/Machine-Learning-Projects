import streamlit as st
import numpy as np
import joblib 
import warnings
warnings.filterwarnings("ignore")


model = joblib.load(r"D:\Robotics\Machine Learning Projects\Exam score prediction\From YT\src\model.plk")
st.title("Exam score prediction")

study_hours = st.slider("study hours per day" , 0.0,12.0,1.0)
attendance=st.slider("attendace Percentage",0.0,100.0,80.0)
mental_health = st.slider("mental health rating",1,10,5)
sleep_hours = st.slider("Sleep hours",0.0,12.0,7.0)
part_time_job = st.selectbox("part-time job",["No","Yes"])

ptj_encoded = 1 if part_time_job =="Yes" else 0

if st.button("Predict exam score"):
    input_data = np.array([study_hours, attendance, mental_health,
       ptj_encoded, sleep_hours]).reshape(1,-1)
    prediction = model.predict(input_data)[0]
    prediction= max(0,min(100,prediction))
    st.success(f"Predicted Exam score :{prediction:.2f}")