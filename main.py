# In main.py (root directory)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
from src.data_processing import preprocess_data, engineer_features
from src.model import build_and_train_model, save_model
from src.recommendations import HealthRecommender


def main():
    print("Health Recommendation System - Model Training")

    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Generate synthetic data for demonstration
    print("Generating synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'daily_steps': np.random.normal(7000, 3000, n_samples),
        'exercise_minutes_week': np.random.normal(150, 100, n_samples),
        'water_intake_liters': np.random.normal(1.5, 0.7, n_samples),
        'stress_level': np.random.randint(1, 11, n_samples),
        'nutrition_quality': np.random.randint(1, 11, n_samples),
        'screen_time_hours': np.random.normal(4, 2, n_samples)
    })

    # Generate target variable based on features (simplified for demonstration)
    data['wellbeing_score'] = (
            70 +  # Base score
            (data['sleep_hours'] - 7) * 3 +  # Sleep impact
            (data['exercise_minutes_week'] / 150) * 10 +  # Exercise impact
            (data['water_intake_liters'] - 1.5) * 5 +  # Hydration impact
            (5 - data['stress_level']) * 2 +  # Stress impact (lower is better)
            (data['nutrition_quality'] - 5) * 2  # Nutrition impact
    )
    # Add some noise to make it more realistic
    data['wellbeing_score'] += np.random.normal(0, 5, n_samples)
    # Clip to realistic range
    data['wellbeing_score'] = data['wellbeing_score'].clip(0, 100)

    # Save the synthetic data
    data.to_csv('data/health_data.csv', index=False)
    print(f"Data saved to data/health_data.csv")

    # Engineer features
    print("Engineering features...")
    enhanced_data = engineer_features(data)

    # Preprocess data
    print("Preprocessing data...")
    X, y, preprocessor = preprocess_data(enhanced_data)

    # Train model
    print("Training model...")
    model_pipeline, metrics = build_and_train_model(X, y, preprocessor, model_type='random_forest')

    # Save the model
    save_model(model_pipeline, 'models/health_recommender_model.pkl')

    print("\nModel training complete!")
    print("Run 'python src/app.py' to use the recommendation system.")


if __name__ == "__main__":
    main()