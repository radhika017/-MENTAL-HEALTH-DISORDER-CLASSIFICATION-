import streamlit as st
import pandas as pd
from joblib import load
import pickle
import sklearn

# Load the trained model
model = load('vs_ensemble_trained_model_mhdc.joblib')
#with open('E:\\Winnovation\\projects\\mhdc\\ensemble_trained_model_mhdc.pkl', 'rb') as f:
    #model = pickle.load(f)

#model = pickle.load(open('E:\\Winnovation\\projects\\mhdc\\ensemble_trained_model_mhdc.pkl', 'rb'))

# Define the Streamlit app
def main():
    st.title('Mental Health Disorder Classifier')
    
    # Add input widgets for user input
    st.sidebar.header('User Input')
    sadness = st.sidebar.slider('Sadness', 0, 10, 5)
    euphoric = st.sidebar.slider('Euphoric', 0, 10, 5)
    exhausted = st.sidebar.slider('Exhausted', 0, 10, 5)
    sleep_disorder = st.sidebar.radio('Sleep Disorder', ['Yes', 'No'])
    mood_swing = st.sidebar.radio('Mood Swing', ['Yes', 'No'])
    suicidal_thoughts = st.sidebar.radio('Suicidal Thoughts', ['Yes', 'No'])
    anorexia = st.sidebar.radio('Anorexia', ['Yes', 'No'])
    authority_respect = st.sidebar.radio('Authority Respect', ['Yes', 'No'])
    try_explanation = st.sidebar.radio('Try Explanation', ['Yes', 'No'])
    aggressive_response = st.sidebar.radio('Aggressive Response', ['Yes', 'No'])
    ignore_move_on = st.sidebar.radio('Ignore & Move On', ['Yes', 'No'])
    nervous_breakdown = st.sidebar.radio('Nervous Breakdown', ['Yes', 'No'])
    admit_mistakes = st.sidebar.radio('Admit Mistakes', ['Yes', 'No'])
    overthinking = st.sidebar.radio('Overthinking', ['Yes', 'No'])
    sexual_activity = st.sidebar.slider('Sexual Activity', 0, 10, 5)
    concentration = st.sidebar.slider('Concentration', 0, 10, 5)
    optimism = st.sidebar.slider('Optimism', 0, 10, 5)
    
    # Create a dataframe with user input
    user_input = pd.DataFrame({
        'Sadness': [sadness],
        'Euphoric': [euphoric],
        'Exhausted': [exhausted],
        'Sleep dissorder': [1 if sleep_disorder == 'Yes' else 0],
        'Mood Swing': [1 if mood_swing == 'Yes' else 0],
        'Suicidal thoughts': [1 if suicidal_thoughts == 'Yes' else 0],
        'Anorxia': [1 if anorexia == 'Yes' else 0],
        'Authority Respect': [1 if authority_respect == 'Yes' else 0],
        'Try-Explanation': [1 if try_explanation == 'Yes' else 0],
        'Aggressive Response': [1 if aggressive_response == 'Yes' else 0],
        'Ignore & Move-On': [1 if ignore_move_on == 'Yes' else 0],
        'Nervous Break-down': [1 if nervous_breakdown == 'Yes' else 0],
        'Admit Mistakes': [1 if admit_mistakes == 'Yes' else 0],
        'Overthinking': [1 if overthinking == 'Yes' else 0],
        'Sexual Activity': [sexual_activity],
        'Concentration': [concentration],
        'Optimisim': [optimism]
    })
    
    # Make prediction
    prediction_proba = model.predict_proba(user_input)
    reverse_mapping = {0: 'Normal', 1: 'Bipolar Type-1', 2: 'Bipolar Type-2', 3: 'Depression'}
    #predicted_class = reverse_mapping[predicted_label]

    # Display the predicted probabilities
    st.subheader('Prediction Probabilities')
    st.write(pd.DataFrame({
        
        'Class': model.classes_,
        'Disorder':['Normal','Bipolar Type-1','Bipolar Type-2','Depression'],
        'Probability': prediction_proba[0]
    }))

if __name__ == '__main__':
    main()
