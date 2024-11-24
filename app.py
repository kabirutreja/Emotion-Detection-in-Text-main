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
        background: linear-gradient(90deg, #ff7eb3, #ff758c, #ff6f61, #ff5252);
        background-size: 200% 200%;
        animation: gradient 4s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    button {
        font-size: 16px;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    button:hover {
        background-color: #ffa3a3;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Emotion Classifier App 🌟")
    st.subheader("Analyze and understand your emotions with AI 🔍")

    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Enter your text below:", placeholder="Type something here...")
        submit_text = st.form_submit_button(label='Analyze 🚀')

    if submit_text:
        with st.spinner("Analyzing..."):
            if not raw_text.strip():
                st.error("❌ Please enter some text to analyze!")
                return

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

            col1, col2 = st.columns(2)
            with col1:
                st.success("🔤 Input Text")
                st.write(raw_text)
                st.success("📊 Detected Emotion")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"{prediction.capitalize()}: {emoji_icon}")
                st.write(f"Confidence Score: {np.max(probability):.2f}")

            with col2:
                st.success("📈 Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x=alt.X('Emotions', sort=None),
                    y='Probability',
                    color=alt.Color('Emotions', scale=alt.Scale(scheme="spectral"))
                ).properties(width='container', height=300)
                st.altair_chart(fig, use_container_width=True)

def render_monitor():
    add_page_visited_details("Monitor", datetime.now(IST))
    st.title("📊 App Monitoring and Metrics")

    page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Visit Time'])
    df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Text', 'Prediction', 'Confidence', 'Timestamp'])

    with st.expander("📅 Page Metrics"):
        st.dataframe(page_visited_details)
        pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
        p = px.pie(pg_count, values='Counts', names='Page Name', color_discrete_sequence=px.colors.sequential.Brwnyl)
        st.plotly_chart(p, use_container_width=True)

    with st.expander('🧠 Emotion Classifier Metrics'):
        st.dataframe(df_emotions)

def render_about():
    add_page_visited_details("About", datetime.now(IST))
    st.title("💡 About This App")
    st.markdown("""
        This **Emotion Classifier App** leverages AI to understand the emotions hidden in text. 🚀

        ### Features
        - **Real-time Emotion Analysis**
        - **Confidence Scores**
        - **Interactive Visuals**
    """)

def main():
   
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    with st.sidebar:
        st.image("./logo.png", width=200)
        st.header("🛠 Navigation")
        for page in ["Home", "Monitor", "About"]:
            if st.button(page, key=page):
                st.session_state.current_page = page

    if st.session_state.current_page == "Home":
        render_home()
    elif st.session_state.current_page == "Monitor":
        render_monitor()
    elif st.session_state.current_page == "About":
        render_about()

if __name__ == '__main__':
    main()
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
