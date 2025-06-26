# In src/app.py
import pandas as pd
import joblib
import os
from recommendations import HealthRecommender


def load_model(model_path):
    """Load the trained model."""
    return joblib.load(model_path)


def run_console_app():
    """Run the health recommendation system as a console application."""
    print("\n===== Health Lifestyle Recommendation System =====\n")

    # Load the model
    model_path = '../models/health_recommender_model.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = load_model(model_path)
    recommender = HealthRecommender(model)

    # Collect user input
    print("Please provide your health information:")
    user_data = {}

    try:
        user_data['age'] = int(input("Age: "))
        user_data['gender'] = input("Gender (M/F): ").upper()
        user_data['sleep_hours'] = float(input("Average sleep hours per night: "))
        user_data['daily_steps'] = int(input("Average daily steps: "))
        user_data['exercise_minutes_week'] = int(input("Exercise minutes per week: "))
        user_data['water_intake_liters'] = float(input("Water intake in liters per day: "))
        user_data['stress_level'] = int(input("Stress level (1-10, where 10 is highest): "))
        user_data['nutrition_quality'] = int(input("Nutrition quality (1-10, where 10 is best): "))
        user_data['screen_time_hours'] = float(input("Screen time hours per day: "))
    except ValueError:
        print("\nError: Please enter valid numerical values.")
        return

    # Generate recommendations
    result = recommender.generate_recommendations(user_data)

    # Display results
    print("\n===== Your Health Analysis =====\n")
    print(f"Predicted Wellbeing Score: {result['predicted_wellbeing']:.2f}/100")

    print("\n===== Personalized Recommendations =====\n")

    if result['priority_areas']:
        print("Priority Areas for Improvement:")
        for area in result['priority_areas']:
            feature_name = area.replace('_', ' ').title()
            print(f"• {feature_name}: {result['recommendations'][area]['recommendation']}")

    print("\nAll Recommendations:")
    for feature, details in result['recommendations'].items():
        feature_name = feature.replace('_', ' ').title()
        print(f"• {feature_name} ({details['category']}): {details['recommendation']}")

    print("\nThank you for using the Health Lifestyle Recommendation System!")


if __name__ == "__main__":
    run_console_app()