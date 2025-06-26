# Example Streamlit app
import streamlit as st
import pandas as pd
import joblib
from src.recommendations import HealthRecommender

st.title("Health Lifestyle Recommendation System")

# Load model
model = joblib.load('models/health_recommender_model.pkl')
recommender = HealthRecommender(model)

# Create input form
st.header("Enter Your Health Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", options=["M", "F"])
    sleep_hours = st.slider("Sleep Hours per Night", 3.0, 12.0, 7.0, 0.5)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000)
    exercise_minutes = st.slider("Exercise Minutes per Week", 0, 600, 150, 10)

with col2:
    water_intake = st.slider("Water Intake (Liters per Day)", 0.0, 5.0, 1.5, 0.1)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    nutrition_quality = st.slider("Nutrition Quality (1-10)", 1, 10, 5)
    screen_time = st.slider("Screen Time (Hours per Day)", 0.0, 16.0, 4.0, 0.5)

if st.button("Get Recommendations"):
    user_data = {
        'age': age,
        'gender': gender,
        'sleep_hours': sleep_hours,
        'daily_steps': daily_steps,
        'exercise_minutes_week': exercise_minutes,
        'water_intake_liters': water_intake,
        'stress_level': stress_level,
        'nutrition_quality': nutrition_quality,
        'screen_time_hours': screen_time
    }

    results = recommender.generate_recommendations(user_data)

    st.header("Your Health Analysis")
    st.subheader(f"Wellbeing Score: {results['predicted_wellbeing']:.1f}/100")

    st.header("Personalized Recommendations")

    if results['priority_areas']:
        st.subheader("Priority Areas for Improvement")
        for area in results['priority_areas']:
            feature_name = area.replace('_', ' ').title()
            st.info(f"**{feature_name}**: {results['recommendations'][area]['recommendation']}")

    st.subheader("All Recommendations")
    for feature, details in results['recommendations'].items():
        feature_name = feature.replace('_', ' ').title()
        st.write(f"**{feature_name}**: {details['recommendation']}")