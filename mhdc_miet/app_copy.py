import streamlit as st
import pandas as pd
from joblib import load
from fpdf import FPDF, HTMLMixin
import tempfile

# Load the trained model
model = load('vs_ensemble_trained_model_mhdc.joblib')

class PDF(FPDF, HTMLMixin):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Model institute of Engineering and Technology', 0, 1, 'C')
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Mental Health Disorder Report', 0, 1, 'C')
        self.image('MIET.png', 10, 8, 20)
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

# Define the Streamlit app
def main():
    st.title('Mental Health Disorder Classifier')
    
    # Add input widgets for user information
    st.sidebar.header('User Information')
    name = st.sidebar.text_input('Name')
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=25)
    
    # Add input widgets for symptoms
    st.sidebar.header('Symptom Input')
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
    predicted_class = reverse_mapping[prediction_proba.argmax()]

    # Display the predicted probabilities
    st.subheader('Prediction Probabilities')
    prediction_df = pd.DataFrame({
        'Class': model.classes_,
        'Disorder': ['Normal','Bipolar Type-1','Bipolar Type-2','Depression'],
        'Probability': prediction_proba[0]
    })
    st.write(prediction_df)

    # Display suggestions based on predictions
    st.subheader('Suggestions')
    if predicted_class == 'Normal':
        st.write("It seems that you're doing well. Continue maintaining a healthy lifestyle.")
    elif predicted_class == 'Bipolar Type-1':
        st.write("Consider seeking professional help and maintaining a routine to manage mood swings. Medication and therapy can be very effective.")
    elif predicted_class == 'Bipolar Type-2':
        st.write("Monitor your mood and consider therapy to manage symptoms effectively. Medication can also help in stabilizing mood swings.")
    elif predicted_class == 'Depression':
        st.write("It is recommended to seek professional counseling or therapy and discuss potential treatment options. Medications and lifestyle changes can make a significant difference.")

    # Generate PDF report
    if st.button('Generate PDF Report'):
        generate_pdf_report(name, age, user_input, prediction_df, predicted_class)

def generate_pdf_report(name, age, user_input, prediction_df, predicted_class):
    pdf = PDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', size=16)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(200, 10, txt="Mental Health Disorder Report", ln=True, align='C')

    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Model institute of Engineering and Technology", ln=True, align='C')

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=10)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Project by: Radhika and Sonika", ln=True, align='C')
    pdf.cell(200, 10, txt="Model Used: Deep Learning ", ln=True, align='C')
    pdf.image('MIET.png', x=10, y=8, w=20)
    
    pdf.set_font("Arial", size=12)
    pdf.ln(20)
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.set_fill_color(224, 235, 255)
    pdf.cell(200, 10, txt="Symptom Input:", ln=True, fill=True)
    for column in user_input.columns:
        pdf.cell(200, 10, txt=f"{column}: {user_input[column][0]}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.set_fill_color(224, 235, 255)
    pdf.cell(200, 10, txt="Prediction Probabilities:", ln=True, fill=True)
    for i in range(len(prediction_df)):
        disorder = prediction_df['Disorder'][i]
        probability = prediction_df['Probability'][i]
        pdf.cell(200, 10, txt=f"{disorder}: {probability*100:.2f}%", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.set_fill_color(224, 235, 255)
    if predicted_class == 'Normal':
        pdf.set_text_color(0, 128, 0)
        pdf.cell(200, 10, txt=f"Predicted Disorder: {predicted_class}", ln=True, fill=True)
    else:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, txt=f"Predicted Disorder: {predicted_class}", ln=True, fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.set_fill_color(224, 235, 255)
    pdf.cell(200, 10, txt="Suggestions:", ln=True, fill=True)
    pdf.set_font("Arial", size=12)
    if predicted_class == 'Normal':
                pdf.multi_cell(0, 10, "It seems that you're doing well. Continue maintaining a healthy lifestyle.")
    elif predicted_class == 'Bipolar Type-1':
        pdf.multi_cell(0, 10, "Consider seeking professional help and maintaining a routine to manage mood swings. Medication and therapy can be very effective.")
    elif predicted_class == 'Bipolar Type-2':
        pdf.multi_cell(0, 10, "Monitor your mood and consider therapy to manage symptoms effectively. Medication can also help in stabilizing mood swings.")
    elif predicted_class == 'Depression':
        pdf.multi_cell(0, 10, "It is recommended to seek professional counseling or therapy and discuss potential treatment options. Medications and lifestyle changes can make a significant difference.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)

    with open(tmpfile.name, "rb") as f:
        st.download_button(
            label="Download PDF Report",
            data=f,
            file_name="mental_health_report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
