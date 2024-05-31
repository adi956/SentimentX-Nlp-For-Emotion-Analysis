import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Use double backslashes or a raw string for file paths
model_path = r"D:\Projects\SentimentX\model\text_emotion.pkl"

# Load the model
pipe_lr = joblib.load(open(model_path, "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.set_page_config(page_title="SentimentX", page_icon="😊", layout="wide")
    
    # Custom CSS for dark mode
    st.markdown("""
        <style>
        .main {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        .title {
            color: #ff4b4b;
            font-size: 3em;
        }
        .subheader {
            color: #ff7f50;
            font-size: 2em;
        }
        .stTextArea, .stForm {
            color: #000000;  /* Text area input text should remain dark for better readability */
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("SentimentX")
    st.subheader("Emotions Analysis of Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here", placeholder="Enter your text here...")
        submit_text = st.form_submit_button(label='Analyze')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.markdown(f"<h2 style='color:#ff4b4b;'>{prediction} {emoji_icon}</h2>", unsafe_allow_html=True)
            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X('emotions', sort=None),
                y='probability',
                color=alt.Color('emotions', scale=alt.Scale(scheme='dark2')),
                tooltip=['emotions', 'probability']
            ).properties(
                width=600,
                height=400,
                background='#1e1e1e',
                padding={"left": 5, "right": 5, "top": 5, "bottom": 5}
            ).configure_axis(
                labelColor='#ffffff',
                titleColor='#ffffff'
            ).configure_legend(
                labelColor='#ffffff',
                titleColor='#ffffff'
            ).configure_title(
                color='#ffffff'
            ).interactive()

            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
