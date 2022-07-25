import streamlit as st
import boto3
from voicerecognizer import voiceToText 


st.title('Sentimental Analaysis')

if st.button("predict"):
    st.markdown('Recording started : ')
    text = voiceToText()
    st.markdown("you spoke : "+str(text))
    client = boto3.client('comprehend')
    response = client.detect_sentiment(Text=text, LanguageCode='en')

    if response["Sentiment"] == "POSITIVE":
        st.success("Positive Statement")
    elif response["Sentiment"] == "NEGATIVE":
        st.success("Negative Statement")
    elif response["Sentiment"] == "NEUTRAL":
        st.success("Neutral Statement")
    elif response["Sentiment"] == "MIXED":
        st.success("Mixed Statement")
    else:
        st.error("Invalid Text")
