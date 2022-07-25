import streamlit as st
import boto3

st.title("TextDetection Using AWS")

img_file=st.file_uploader("Upload a file",type=['png','jpg','jpeg'])

if img_file is not None:
    
    with open('input.jpg','wb') as f:
        f.write(img_file.getbuffer())
        client = boto3.client('rekognition')
        imageSource=open('input.jpg','rb')
        respone = client.detect_text(
            Image={'Bytes':imageSource.read()}
        )
        if(len(respone['TextDetections'])>0):
            st.success("Text detected")
            for text in respone['TextDetections']:
                st.markdown(text['DetectedText'])
        else:
            st.error("No Text Detected")