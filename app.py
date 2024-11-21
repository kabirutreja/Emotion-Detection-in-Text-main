import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from datetime import datetime
import joblib
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST

# Load Model
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

# Functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Emoji dictionary
emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Page Rendering Functions
def render_home():
    add_page_visited_details("Home", datetime.now(IST))
    st.markdown("""
    <style>
    h1 {
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        background: linear-gradient(90deg, #00a3cc, #00b8e6, #00ccff, #1ad1ff);
        background-size: 200% 200%;
        animation: gradient 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
""", unsafe_allow_html=True)


    st.title("Emotion Classifier App")
    st.subheader("Analyze the Emotions in Your Text")

    # Form for emotion classification
    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Enter your text below:")
        submit_text = st.form_submit_button(label='Analyze')

    if submit_text:
        if not raw_text.strip():
            st.error("Please enter some text to analyze.")
            return

        # Perform prediction
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        # Save prediction details
        add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.success("Input Text")
            st.write(raw_text)
            st.success("Detected Emotion")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(f"{prediction.capitalize()}: {emoji_icon}")
            st.write(f"Confidence Score: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotions", "Probability"]

            # Create Altair bar chart
            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X('Emotions', sort=None),
                y='Probability',
                color=alt.Color('Emotions', scale=alt.Scale(scheme="blues"))
            ).properties(width='container', height=300)
            st.altair_chart(fig, use_container_width=True)

def render_monitor():
    add_page_visited_details("Monitor", datetime.now(IST))
    st.title("App Monitoring and Metrics")

    # Preload data for faster loading
    page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Visit Time'])
    df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Text', 'Prediction', 'Confidence', 'Timestamp'])

    # Display page visit metrics
    with st.expander("Page Metrics"):
        st.dataframe(page_visited_details)

        # Pie chart using Plotly
        pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
        p = px.pie(pg_count, values='Counts', names='Page Name', color_discrete_sequence=px.colors.sequential.dense)
        st.plotly_chart(p, use_container_width=True)

    # Display emotion prediction metrics
    with st.expander('Emotion Classifier Metrics'):
        st.dataframe(df_emotions)

def render_about():
    add_page_visited_details("About", datetime.now(IST))
    st.title("About This App")
    st.markdown("""
        Welcome to the Emotion Classifier App! This tool is designed to analyze text for underlying emotions using advanced machine learning techniques.

        ### Features
        - Real-time Emotion Detection
        - Confidence Scores for predictions
        - User-friendly Interface

        ### Applications
        - **Sentiment Analysis** for social media and customer feedback
        - **Market Research** for consumer insights
        - **Content Analysis** for brand monitoring
    """)

# Main Application
def main():
    st.set_page_config(page_title="Emotion Classifier App", layout="wide")

    # Set up session state for page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    # Sidebar Navigation
    with st.sidebar:
        st.image("./logo.png", width=200)  # Add your logo here
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        st.header("Navigation")
        pages = ["Home", "Monitor", "About"]
        for page in pages:
            if st.button(page, key=page):
                st.session_state.current_page = page

    # Render selected page
    if st.session_state.current_page == "Home":
        render_home()
    elif st.session_state.current_page == "Monitor":
        render_monitor()
    elif st.session_state.current_page == "About":
        render_about()

if __name__ == '__main__':
    main()
