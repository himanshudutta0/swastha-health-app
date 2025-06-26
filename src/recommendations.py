# In src/recommendations.py
import pandas as pd
import numpy as np


class HealthRecommender:
    def __init__(self, model):
        """
        Initialize the health recommendation system.

        Parameters:
        -----------
        model : sklearn.pipeline.Pipeline
            Trained ML model
        """
        self.model = model

        # Define recommendation rules
        self.recommendation_rules = {
            'sleep_hours': {
                'low': 'Try to increase your sleep to 7-9 hours per night. Set a consistent sleep schedule and limit screen time before bed.',
                'optimal': 'Great job maintaining healthy sleep habits! Keep up the 7-9 hour sleep schedule.',
                'high': 'You might be oversleeping. While sleep is important, too much can be associated with health issues. Aim for 7-9 hours.'
            },
            'exercise_minutes_week': {
                'low': 'Try to increase physical activity to at least 150 minutes per week. Even short walks help!',
                'optimal': "You're meeting the recommended exercise guidelines. Keep up the good work!",
                'high': 'You have an active lifestyle! Make sure to include rest days for recovery.'
            },
            'water_intake_liters': {
                'low': 'Try to increase your water intake to at least 2 liters per day for proper hydration.',
                'optimal': "You're staying well-hydrated. Keep it up!",
                'high': "Your water intake is high, which is generally good, but make sure it's not excessive."
            },
            'advice': {
                'stress_level': {
                    'high': 'Your stress levels appear high. Consider stress-reduction techniques like meditation, yoga, or deep breathing.',
                    'moderate': 'You are managing stress reasonably well. Regular relaxation techniques can help maintain this.',
                    'low': 'You are managing stress well. Continue your effective stress management practices.'
                },
                'nutrition_quality': {
                    'low': 'Focus on improving your diet with more whole foods, fruits, and vegetables.',
                    'moderate': 'Your nutrition is decent. Try to include more variety in your diet for additional benefits.',
                    'high': 'Your nutrition quality is excellent. Maintain your healthy eating habits!'
                },
                'screen_time_hours': {
                    'high': 'Consider reducing screen time and taking regular breaks to prevent eye strain and improve wellbeing.',
                    'moderate': 'Your screen time is moderate. Remember to take breaks and practice the 20-20-20 rule (look at something 20 feet away for 20 seconds every 20 minutes).',
                    'low': 'You are doing well limiting screen time, which benefits both your eyes and overall health.'
                }
            }
        }

    def _categorize_feature(self, feature_name, value):
        """Categorize a feature value into low, moderate, optimal, or high."""
        if feature_name == 'sleep_hours':
            if value < 7:
                return 'low'
            elif 7 <= value <= 9:
                return 'optimal'
            else:
                return 'high'
        elif feature_name == 'exercise_minutes_week':
            if value < 150:
                return 'low'
            elif 150 <= value <= 300:
                return 'optimal'
            else:
                return 'high'
        elif feature_name == 'water_intake_liters':
            if value < 2:
                return 'low'
            elif 2 <= value <= 3:
                return 'optimal'
            else:
                return 'high'
        elif feature_name == 'stress_level':
            if value <= 3:
                return 'low'
            elif 4 <= value <= 7:
                return 'moderate'
            else:
                return 'high'
        elif feature_name == 'nutrition_quality':
            if value <= 4:
                return 'low'
            elif 5 <= value <= 7:
                return 'moderate'
            else:
                return 'high'
        elif feature_name == 'screen_time_hours':
            if value <= 2:
                return 'low'
            elif 2 < value <= 4:
                return 'moderate'
            else:
                return 'high'
        return 'moderate'  # Default fallback

    def predict_wellbeing(self, user_data):
        """
        Predict wellbeing score based on user data.

        Parameters:
        -----------
        user_data : pandas.DataFrame
            User health and lifestyle data (single row)

        Returns:
        --------
        float
            Predicted wellbeing score
        """
        if isinstance(user_data, dict):
            user_data = pd.DataFrame([user_data])

        return self.model.predict(user_data)[0]

    def generate_recommendations(self, user_data):
        """
        Generate personalized health recommendations.

        Parameters:
        -----------
        user_data : dict or pandas.DataFrame
            User health and lifestyle data

        Returns:
        --------
        dict
            Personalized recommendations and predicted wellbeing score
        """
        if isinstance(user_data, dict):
            user_df = pd.DataFrame([user_data])
        else:
            user_df = user_data.copy()

        # Predict wellbeing score
        predicted_score = self.predict_wellbeing(user_df)

        # Generate recommendations based on feature values
        recommendations = {}
        priority_areas = []

        for feature in self.recommendation_rules.keys():
            if feature in user_df.columns:
                value = user_df[feature].iloc[0]
                category = self._categorize_feature(feature, value)

                # Only add recommendations for areas that need improvement
                if category in self.recommendation_rules[feature]:
                    recommendation = self.recommendation_rules[feature][category]
                    recommendations[feature] = {
                        'value': value,
                        'category': category,
                        'recommendation': recommendation
                    }

                    # Identify priority areas (those with 'low' or 'high' categories that need improvement)
                    if (category == 'low' and feature != 'stress_level') or \
                            (category == 'high' and feature in ['stress_level', 'screen_time_hours']):
                        priority_areas.append(feature)

        return {
            'predicted_wellbeing': predicted_score,
            'recommendations': recommendations,
            'priority_areas': priority_areas
        }