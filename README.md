# Health Lifestyle Recommendation System

A machine learning system that provides personalized health recommendations based on lifestyle data.

## Project Overview

This project uses machine learning to analyze health and lifestyle data and generate personalized wellness recommendations. It demonstrates the following ML skills:

- Data preprocessing and feature engineering
- Model training and hyperparameter tuning
- Building a recommendation system
- Creating a simple user interface

## Features

- Predicts overall wellbeing scores based on health metrics
- Identifies priority areas for health improvement
- Provides personalized, actionable recommendations
- Uses an ensemble machine learning model
- Handles various types of health data

## Data

The system works with the following health metrics:
- Sleep duration
- Physical activity levels
- Water intake
- Stress levels
- Nutrition quality
- Screen time
- Basic demographics (age, gender)

## Technical Details

- **Languages/Frameworks**: Python, Scikit-learn
- **ML Algorithms**: Random Forest, Gradient Boosting
- **Data Processing**: Feature engineering, scaling, one-hot encoding

## Project Structure
health_recommender/
├── data/                   # Data storage
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data_processing.py  # Data cleaning and preparation
│   ├── feature_engineering.py
│   ├── model.py            # Model definition and training
│   ├── recommendations.py  # Recommendation engine
│   └── app.py              # User interface
├── main.py                 # Main script to run the entire pipeline
└── README.md               # Project documentation

## How to Use

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the training pipeline: `python main.py`
4. Use the recommendation system: `python src/app.py`

## Future Improvements

- Add a web-based interface using Flask or Streamlit
- Incorporate more sophisticated health metrics
- Implement user feedback to improve recommendations over time
- Add data visualization components