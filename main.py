import pandas as pd
import streamlit as st
import pickle

def main():
    st.title("Stress Level Prediction App")

    # Define the correct column names expected by the model
    column_names = [
        'depression', 'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem', 'noise_level',
        'living_conditions', 'safety', 'basic_needs', 'academic_performance', 'study_load', 'teacher_student_relationship',
        'future_career_concerns', 'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying', 'mental_health_history_n', 'self_esteem_n', 'anxiety_level_n']


    test_df = pd.DataFrame(0, index=[0], columns=column_names)

    # Input Fields
    anxiety_level = st.number_input("Anxiety level(0-21):", min_value=0, max_value=21, value=15, key='anxiety_level_key')
    self_esteem = st.number_input("Self Esteem(0-30):", min_value=0, max_value=30, value=15, key='self_esteem_key')
    mental_health_history = st.number_input("Mental Health History (0 or 1):", min_value=0, max_value=1, value=0, key='mental_health_history_key')
    depression = st.number_input("Depression (0-27):", min_value=0, max_value=27, value=10, key='depression_key')
    headache = st.number_input("Headache (0-5):", min_value=0, max_value=5, value=3, key='headache_key')
    blood_pressure = st.number_input("Blood Pressure (0-3):", min_value=0, max_value=3, value=1, key='blood_pressure_key')
    sleep_quality = st.number_input("Sleep Quality (0-5):", min_value=0, max_value=5, value=4, key='sleep_quality_key')
    breathing_problem = st.number_input("Breathing Problem (0-5):", min_value=0, max_value=5, value=2, key='breathing_problem_key')
    noise_level = st.number_input("Noise Level (0-5):", min_value=0, max_value=5, value=2, key='noise_level_key')
    living_conditions = st.number_input("Living Conditions (0-5):", min_value=0, max_value=5, value=4, key='living_conditions_key')
    safety = st.number_input("Safety (0-5):", min_value=0, max_value=5, value=5, key='safety_key')
    basic_needs = st.number_input("Basic Needs (0-5):", min_value=0, max_value=5, value=5, key='basic_needs_key')
    academic_performance = st.number_input("Academic Performance (0-5):", min_value=0, max_value=5, value=4, key='academic_performance_key')
    study_load = st.number_input("Study Load (0-5):", min_value=0, max_value=5, value=3, key='study_load_key')
    teacher_student_relationship = st.number_input("Teacher-Student Relationship (0-5):", min_value=0, max_value=5, value=4, key='teacher_student_relationship_key')
    future_career_concerns = st.number_input("Future Career Concerns (0-5):", min_value=0, max_value=5, value=3, key='future_career_concerns_key')
    social_support = st.number_input("Social Support (0-3):", min_value=0, max_value=3, value=2, key='social_support_key')
    peer_pressure = st.number_input("Peer Pressure (0-5):", min_value=0, max_value=5, value=3, key='peer_pressure_key')
    extracurricular_activities = st.number_input("Extracurricular Activities (0-5):", min_value=0, max_value=5, value=3, key='extracurricular_activities_key')
    bullying = st.number_input("Bullying (0-5):", min_value=0, max_value=5, value=1, key='bullying_key')

    # Assign values to the DataFrame
    data = {
        'anxiety_level_n' :anxiety_level,
        'self_esteem_n': self_esteem,
        'mental_health_history_n': mental_health_history,
        'depression': depression,  # Adjusted the name to match the model's expected input
        'headache': headache,
        'blood_pressure': blood_pressure,
        'sleep_quality': sleep_quality,
        'breathing_problem': breathing_problem,
        'noise_level': noise_level,
        'living_conditions': living_conditions,
        'safety': safety,
        'basic_needs': basic_needs,
        'academic_performance': academic_performance,
        'study_load': study_load,
        'teacher_student_relationship': teacher_student_relationship,
        'future_career_concerns': future_career_concerns,
        'social_support': social_support,
        'peer_pressure': peer_pressure,
        'extracurricular_activities': extracurricular_activities,
        'bullying': bullying
    }

    for col, value in data.items():
        test_df[col] = value

    # Load the trained model and make a prediction
    try:
        with open('xgboost_model.pkl', 'rb') as file:
            clf = pickle.load(file)

        # Predict stress level
        prediction = clf.predict(test_df)

        # Display the prediction result
        st.success(f"Predicted Stress Level: {prediction[0]}")
    except FileNotFoundError:
        st.error("The model file 'xgboost_model.pkl' was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
