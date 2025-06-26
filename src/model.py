# In src/model.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np
import joblib


def build_and_train_model(X, y, preprocessor, model_type='random_forest', random_state=42):
    """
    Build and train a health recommendation model.

    Parameters:
    -----------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target variable
    preprocessor : sklearn.compose.ColumnTransformer
        Fitted preprocessor for transforming data
    model_type : str
        Type of model to train ('random_forest', 'gradient_boosting', or 'linear')
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    sklearn.pipeline.Pipeline
        Trained model pipeline
    dict
        Model evaluation metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Select model type
    if model_type == 'random_forest':
        model = RandomForestRegressor(random_state=random_state)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10]
        }
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=random_state)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    elif model_type == 'linear':
        model = LinearRegression()
        param_grid = {}  # Linear regression doesn't have these hyperparameters
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train the model with hyperparameter tuning if applicable
    if param_grid:
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        best_pipeline = pipeline
        best_pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = best_pipeline.predict(X_test)
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    return best_pipeline, metrics


def save_model(model, filepath):
    """Save the trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a trained model from disk."""
    return joblib.load(filepath)