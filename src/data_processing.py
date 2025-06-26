import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(data):
    """
    Preprocess the health data for model training.

    Parameters:
    -----------
    data : pandas.DataFrame
        Raw health and lifestyle data

    Returns:
    --------
    X : pandas.DataFrame
        Processed features
    y : pandas.Series
        Target variable (wellbeing_score)
    preprocessor : sklearn.compose.ColumnTransformer
        Fitted preprocessor for transforming new data
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()

    # Handle missing values (if any)
    numeric_features = ['age', 'sleep_hours', 'daily_steps', 'exercise_minutes_week',
                        'water_intake_liters', 'stress_level', 'nutrition_quality',
                        'screen_time_hours']
    categorical_features = ['gender']
    # Fill missing values for numeric features with median
    for feature in numeric_features:
        if df[feature].isnull().sum() > 0:
            df[feature] = df[feature].fillna(df[feature].median())

    # Fill missing values for categorical features with mode
    for feature in categorical_features:
        if df[feature].isnull().sum() > 0:
            df[feature] = df[feature].fillna(df[feature].mode()[0])

    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Prepare features and target
    X = df.drop('wellbeing_score', axis=1)
    y = df['wellbeing_score']

    return X, y, preprocessor


def engineer_features(data):
    """
    Create new features from existing data.

    Parameters:
    -----------
    data : pandas.DataFrame
        Health and lifestyle data

    Returns:
    --------
    pandas.DataFrame
        Data with engineered features
    """
    df = data.copy()

    # Calculate sleep adequacy (based on recommended 7-9 hours)
    df['sleep_adequacy'] = df['sleep_hours'].apply(
        lambda x: 1 if 7 <= x <= 9 else 0)

    # Calculate activity level category
    df['activity_level'] = pd.cut(
        df['exercise_minutes_week'],
        bins=[0, 75, 150, 300, float('inf')],
        labels=['sedentary', 'light', 'moderate', 'vigorous']
    )

    # Calculate hydration adequacy
    df['hydration_adequacy'] = df['water_intake_liters'].apply(
        lambda x: 1 if x >= 2 else 0)

    # Calculate screen time category
    df['screen_time_category'] = pd.cut(
        df['screen_time_hours'],
        bins=[0, 2, 4, 8, float('inf')],
        labels=['minimal', 'moderate', 'high', 'excessive']
    )

    # Calculate overall lifestyle score (custom metric)
    df['lifestyle_score'] = (
            df['sleep_adequacy'] * 25 +
            (df['exercise_minutes_week'] / 300).clip(0, 1) * 25 +
            (df['water_intake_liters'] / 3).clip(0, 1) * 15 +
            (1 - df['stress_level'] / 10) * 20 +
            (df['nutrition_quality'] / 10) * 15
    )

    # Convert categorical features to dummy variables
    if 'activity_level' in df.columns and df['activity_level'].dtype.name == 'category':
        activity_dummies = pd.get_dummies(df['activity_level'], prefix='activity')
        df = pd.concat([df, activity_dummies], axis=1)
        df.drop('activity_level', axis=1, inplace=True)

    if 'screen_time_category' in df.columns and df['screen_time_category'].dtype.name == 'category':
        screen_dummies = pd.get_dummies(df['screen_time_category'], prefix='screen')
        df = pd.concat([df, screen_dummies], axis=1)
        df.drop('screen_time_category', axis=1, inplace=True)

    return df